import logging
import math

import numpy as np
import torch
import torch.distributed as dist

# from . import default_hooks as default
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as default


class UnbiasedRandomBlockState(object):
    r"""
    Stores both the algorithm's hyperparameters and the internal state for all the gradients during the training.
    
    1. compression_ratio: percentage of remaining entires after gradient compression. For example, if gradient has 100 entries
    and compression_ratio = 0.1, then we will get a 10-dimension vector after gradient compression.

    2. ``start_compression_iter`` defers gradient compression until step ``start_compression_iter``, and vanilla allreduce 
    runs prior to step ``start_compression_iter``. This hybrid scheme of **vanilla allreduce + gradient compression** can 
    effectively improve the accuracy. This is because that, the beginning of training phase is usually very sensitive to 
    inaccurate gradients, and compressing gradients too early may make the training quickly take a suboptimal trajectory, 
    which can result in an irrecoverable impact on the accuracy.

    
    Compression statistics are logged every ``compression_stats_logging_frequency`` iterations once PowerSGD compression starts.

    .. warning ::
        If error feedback is enabled, the minimum value of ``start_compression_iter`` allowed in DDP is 2.
        This is because there is another internal optimization that rebuilds buckets at iteration 1 in DDP,
        and this can conflict with any tensor memorized before the rebuild process.
    """  # noqa: B950

    __slots__ = [
        "process_group",
        # The fields below are the hyperparameters that often need to be tuned by the user.
        "compression_ratio",
        "start_compression_iter",
        # The fields below are the hyperparameters that seldom need be tuned by the user.
        # The fields below are internal state.
        "rng",
        "iter",
        # The fields below are for recording compression stats.
        "compression_stats_logging_frequency",
        "next_stats_report",
        "compressed_tensor_memory",
    ]

    def __init__(
        self,
        process_group,
        compression_ratio=0.1,
        start_compression_iter=1_000,
        random_seed=42,
        compression_stats_logging_frequency=10_000,
    ):
        self.process_group = process_group
        self.compression_ratio = compression_ratio
        # Deferring gradient compression util step 'start_compression_iter' can have two advantages:
        # 1) Gradient compression mitigates the accuracy loss. A simple yet effective way is mixing
        # vanilla allreduce with gradient compression.
        # 2) There is an internal optimization of rebuilding buckets process in DDP,
        # in order to save the memory space.
        # This step takes place after the first iteration.
        # However, this means that the shape of input bucketized tensors is subject to change,
        # which will complicate the implementations of error feedback and warm-up.
        # Running vanilla allreduce in the first few iterations can avoid this complexity.
        if start_compression_iter <= 1:
            raise ValueError(
                "Expect `start_compression_iter` > 1 if `use_error_feedback`, "
                "because gradient compression can only be applied after the first two iterations in DDP."
            )
        self.start_compression_iter = start_compression_iter
        self.rng = np.random.RandomState(random_seed)
        # Since there is only a single state instance for all the input buckets,
        # need to maintain a dictionary that maps each bucket index to the local error.
        # Iteration/step in the training loop.
        self.iter = 0
        self.compression_stats_logging_frequency = max(
            1, compression_stats_logging_frequency
        )
        self.next_stats_report = 0
        self.compressed_tensor_memory = {}


    def maybe_increase_iter(self, bucket):
        # Since bucket 0 is the last bucket to allreduce in an iteration.
        # Only increase `iter` when bucket 0 is processed.
        if bucket.is_the_last_bucket_to_allreduce():
            self.iter += 1

        if self.iter == self.start_compression_iter:
            logging.info(
                "Start to apply PowerSGD after {} iterations.".format(self.iter)
            )


def UnbiasedRandomBlock_hook(state: UnbiasedRandomBlockState, bucket: dist.GradBucket) -> torch.futures.Future:

    process_group = state.process_group
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()

    # The input tensor is a flattened 1D tensor.
    input_tensor = bucket.get_tensor()

    # Run vanilla allreduce in the first `start_compression_iter` iterations.
    if state.iter < state.start_compression_iter:
        state.maybe_increase_iter(bucket)
        return default._allreduce_fut(group_to_use, input_tensor)

    # Apply compression after `start_compression_iter` iterations.
    device = input_tensor.device
    dtype = input_tensor.dtype

    # initialize error dictionaries
    bucket_index = bucket.get_index()

    # Unflatten the input tensor into per-parameter tensors, for layer-wise compression
    tensors = bucket.get_per_parameter_tensors()

    # Step I: Divide all the tensors into two groups,
    # one will be compressed before allreduce and the other will be directly allreduced without compression.
    # Small tensors such as bias are not compressed
    tensors_to_compress, uncompressed_tensors = [], []
    for tensor in tensors:
        no_need_compression = tensor.ndimension() <= 1
        if not no_need_compression:
            tensors_to_compress.append(tensor)
        else:
            uncompressed_tensors.append(tensor)

    # Step II: Handle uncompressed tensors.
    # Allocate contiguous memory for these tensors to allreduce efficiently
    uncompressed_tensors_memory = (
        torch.cat([tensor.view(-1) for tensor in uncompressed_tensors])
        if uncompressed_tensors
        else torch.tensor([], device=device, dtype=dtype)
    )

    # Step III: Handle compressed tensors.
    total_size = 0
    if bucket_index not in state.compressed_tensor_memory:
        for tensor in tensors_to_compress:
            block_size = max(1, int(state.compression_ratio * tensor.nelement()))
            total_size += block_size
        state.compressed_tensor_memory[bucket_index] = torch.empty(total_size, dtype=torch.float32, device=device)

    # gradient compression: note that here we scale the compressed gradient to ensure unbiasedness
    start_idx_list, end_idx_list, block_size_list, prob_list = [], [], [], []
    idx = 0
    for tensor in tensors_to_compress:
        block_size = max(1, int(state.compression_ratio * tensor.nelement()))
        block_size_list.append(block_size)
        prob = block_size/tensor.nelement()
        prob_list.append(prob)
        start_idx = state.rng.choice(tensor.nelement())
        start_idx_list.append(start_idx)
        end_idx = min(start_idx + block_size, tensor.nelement())
        end_idx_list.append(end_idx)
        state.compressed_tensor_memory[bucket_index][idx : idx + end_idx - start_idx].copy_(tensor.view(-1)[start_idx:end_idx])
        state.compressed_tensor_memory[bucket_index][idx: idx + end_idx - start_idx].div_(prob)
        rest = block_size - (end_idx - start_idx)

        if rest > 0:
            state.compressed_tensor_memory[bucket_index][idx + end_idx - start_idx: idx + block_size].copy_(tensor.view(-1)[:rest])
            state.compressed_tensor_memory[bucket_index][idx + end_idx - start_idx: idx + block_size].div_(prob)
        idx += block_size

    # This allreduce is only applied to uncompressed tensors,
    # so it should have been kicked off before the above computation on the compressed tensors to hide more communication costs.
    # However, this somehow requires a separate future chain at this time.
    allreduce_contiguous_uncompressed_tensors_fut = dist.all_reduce(
        uncompressed_tensors_memory, group=group_to_use, async_op=True
    ).get_future()

    def unpack_uncompressed_tensors_and_allreduce_compressed_tensors(fut):
        uncompressed_tensors_memory = fut.value()[0].div_(world_size)
        idx = 0
        for tensor in uncompressed_tensors:
            tensor.copy_(
                uncompressed_tensors_memory[idx : idx + tensor.numel()].view_as(tensor)
            )
            idx += tensor.numel()

        return [
            dist.all_reduce(
                state.compressed_tensor_memory[bucket_index], group=group_to_use, async_op=True
            )
            .get_future()
            .wait()[0]
        ]

    def decompress(fut):
        state.compressed_tensor_memory[bucket_index] = fut.value()[0].div_(world_size)
        # Compute output
        idx = 0
        for tensor, start_idx, end_idx, block_size in zip(tensors_to_compress, start_idx_list, end_idx_list, block_size_list):
            rest = block_size - (end_idx - start_idx)
            tensor.data.zero_()
            values = state.compressed_tensor_memory[bucket_index][idx : idx + block_size]
            tensor.view(-1)[start_idx:end_idx].copy_(values[: end_idx - start_idx])
            if rest > 0:
                tensor.view(-1)[:rest].copy_(values[end_idx - start_idx:])
            idx += block_size

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

        state.maybe_increase_iter(bucket)

        return [input_tensor]

    return (
        allreduce_contiguous_uncompressed_tensors_fut.then(
            unpack_uncompressed_tensors_and_allreduce_compressed_tensors
        )
        .then(decompress)
    )
