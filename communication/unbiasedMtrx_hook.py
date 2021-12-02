
import logging
import math

import numpy as np
import torch
import torch.distributed as dist

from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as default

# from . import default_hooks as default


def _should_compress(
    num_rows, num_cols, matrix_approximation_rank, min_compression_rate
):
    """
    Returns a recommendation as to whether the 2D tensor described by the arguments is worth compressing,
    including statistics describing the expected savings from compression.  We consider a tensor worth
    compressing when ``min_compression_rate`` < uncompressed size / compressed size, where
    uncompressed size = ``num_rows`` * ``num_cols``,
    and compressed size = (``num_rows`` + ``num_cols``) * ``matrix_approximation_rank``.

    The result of this function is a tuple of the form (compression_recommendation, uncompressed_el_count, compressed_el_count), where:

    compresion_recommendation is true if the tensor is worth compressing, and false otherwise (see above);

    uncompressed_el_count is the uncompressed element count, i.e. ``num_rows`` * ``num_cols``; and,

    compress_el_count is the element count after compression, i.e. (``num_rows`` + ``num_cols``) * ``matrix_approximation_rank``.
    """  # noqa: B950
    uncompressed_size = num_rows * num_cols
    compressed_size = (num_rows + num_cols) * matrix_approximation_rank
    return (
        compressed_size * min_compression_rate < uncompressed_size,
        uncompressed_size,
        compressed_size,
    )


def set_random(vector, state, device):
    cur_state = torch.get_rng_state()
    torch.manual_seed(state.rng.randint(1_000_000_000))
    vector.data[:] = torch.randn(*vector.shape, device=device)
    torch.set_rng_state(cur_state)

class unbiasedMtrxState(object):
    """
    This gradient compressor uses random matrices to ensure unbiasedness.
    It has exactly the same communication overhead as powerSGD.
    The numerical performance of this compressor is not great, and it is
    mainly used for comparison purposes.
    """

    __slots__ = [
        "process_group",
        # The fields below are the hyperparameters that often need to be tuned by the user.
        "matrix_approximation_rank",
        "start_powerSGD_iter",
        # The fields below are the hyperparameters that seldom need be tuned by the user.
        "min_compression_rate",
        "orthogonalization_epsilon",
        # The fields below are the binary hyperparameters recommended to be turned on for performance and accuracy.
        # The fields below are internal state.
        "rng",
        "p_memory_dict",
        "q_memory_dict",
        "iter",
        # The fields below are for recording compression stats.
        "total_numel_before_compression",
        "total_numel_after_compression",
        "compression_stats_logging_frequency",
        "next_stats_report",
    ]

    def __init__(
        self,
        process_group,
        matrix_approximation_rank=1,
        start_powerSGD_iter=1_000,
        min_compression_rate=2,
        random_seed=421,
        compression_stats_logging_frequency=10_000,
    ):

        self.process_group = process_group
        self.matrix_approximation_rank = matrix_approximation_rank

        self.start_powerSGD_iter = start_powerSGD_iter
        self.min_compression_rate = min_compression_rate

        self.rng = np.random.RandomState(random_seed)
        # Since there is only a single state instance for all the input buckets,
        # need to maintain a dictionary that maps each bucket index to the local error.
        self.p_memory_dict = {}
        self.q_memory_dict = {}
        # Iteration/step in the training loop.
        self.iter = 0
        # Compression stats accumulators
        self.total_numel_before_compression = 0
        self.total_numel_after_compression = 0
        # We'll report compression stats every 'compression_stats_logging_frequency' iterations
        # Note that we always report compression stats at least once.
        self.compression_stats_logging_frequency = max(
            1, compression_stats_logging_frequency
        )
        self.next_stats_report = 0

    def maybe_increase_iter(self, bucket):
        # Since bucket 0 is the last bucket to allreduce in an iteration.
        # Only increase `iter` when bucket 0 is processed.
        if bucket.is_the_last_bucket_to_allreduce():
            self.iter += 1

        if self.iter == self.start_powerSGD_iter:
            logging.info(
                "Start to apply PowerSGD after {} iterations.".format(self.iter)
            )

    def compression_stats(self):
        r"""
        Returns the latest compression statistics as a tuple of the form (compress_rate, numel_before_compression, numel_after_compression), where:

        compress_rate is the effective compression rate i.e. (number of elements before compression) / (number of elements after compression);

        numel_before_compression is the total number of elements before compression was applied; and,

        numel_after_compression is the total number of elements after compression was applied.
        """  # noqa: B950
        compress_rate = (
            self.total_numel_before_compression / self.total_numel_after_compression
            if self.total_numel_after_compression > 0
            else 0
        )
        return (
            compress_rate,
            self.total_numel_before_compression,
            self.total_numel_after_compression,
        )



def UnbiasedMtrx_hook(
    state: unbiasedMtrxState, bucket: dist.GradBucket
) -> torch.futures.Future:

    process_group = state.process_group
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()

    # The input tensor is a flattened 1D tensor.
    input_tensor = bucket.get_tensor()

    # Run vanilla allreduce in the first `start_powerSGD_iter` iterations.
    if state.iter < state.start_powerSGD_iter:
        state.maybe_increase_iter(bucket)
        return default._allreduce_fut(group_to_use, input_tensor)

    # Apply PowerSGD after `start_powerSGD_iter` iterations.
    device = input_tensor.device
    dtype = input_tensor.dtype

    # Incorporate the error from the previous state into the gradients.
    bucket_index = bucket.get_index()

    total_length = input_tensor.shape[0]

    # Unflatten the input tensor into per-parameter tensors, for layer-wise compression.
    tensors = bucket.get_per_parameter_tensors()

    # Step I: Divide all the tensors into two groups,
    # one will be compressed before allreduce and the other will be directly allreduced without compression.
    tensors_to_compress, uncompressed_tensors = [], []
    total_Ps_size = 0
    total_Qs_size = 0
    for tensor in tensors:
        matrix = tensor.view(tensor.shape[0], -1)
        n, m = matrix.shape
        matrix_approximation_rank = min(n, m, state.matrix_approximation_rank)
        compress_test = _should_compress(
            n, m, matrix_approximation_rank, state.min_compression_rate
        )
        state.total_numel_before_compression += compress_test[1]
        if compress_test[0]:
            tensors_to_compress.append(matrix)
            total_Ps_size += n * matrix_approximation_rank
            total_Qs_size += m * matrix_approximation_rank
            state.total_numel_after_compression += compress_test[2]
        else:
            uncompressed_tensors.append(tensor)
            state.total_numel_after_compression += compress_test[1]

    # _report_compression_stats(bucket, state)

    # Step II: Handle uncompressed tensors.
    # Allocate contiguous memory for these tensors to allreduce efficiently.
    uncompressed_tensors_memory = (
        torch.cat([tensor.view(-1) for tensor in uncompressed_tensors])
        if uncompressed_tensors
        else torch.tensor([], device=device, dtype=dtype)
    )

    # Step III: Handle the tensors that should be compressed.
    # Allocate contiguous memory for Ps and Qs to allreduce efficiently.
    # If warm-start is enabled, reuse Ps and Qs from the previous iteration if possible.
    # The memory spaces of Ps and Qs need to be allocated in the first iteration when PowerSGD is applied.
    if bucket_index not in state.p_memory_dict:
        # If warm-start is disabled, low-rank tensors will be initialized at every step.
        state.p_memory_dict[bucket_index] = torch.empty(
            total_Ps_size, device=device, dtype=dtype
        )
        state.q_memory_dict[bucket_index] = torch.empty(
            total_Qs_size, device=device, dtype=dtype
        )

    # Create Ps and Qs that point to the allocated memory.
    ps = []
    qs = []
    p_idx = 0
    q_idx = 0
    for tensor in tensors_to_compress:
        n, m = tensor.shape
        matrix_approximation_rank = min(n, m, state.matrix_approximation_rank)
        ps.append(
            state.p_memory_dict[bucket_index][
                p_idx : p_idx + n * matrix_approximation_rank
            ].view(n, matrix_approximation_rank)
        )
        qs.append(
            state.q_memory_dict[bucket_index][
                q_idx : q_idx + m * matrix_approximation_rank
            ].view(m, matrix_approximation_rank)
        )
        p_idx += n * matrix_approximation_rank
        q_idx += m * matrix_approximation_rank

    # Compute Ps.
    for tensor, q, p in zip(tensors_to_compress, qs, ps):
        set_random(q, state, device)
        torch.matmul(tensor, q, out=p)

    # This allreduce is only applied to uncompressed tensors,
    # so it should have been kicked off before the above computation on the compressed tensors to hide more communication costs.
    # However, this somehow requires a separate future chain at this time.
    allreduce_contiguous_uncompressed_tensors_fut = dist.all_reduce(
        uncompressed_tensors_memory, group=group_to_use, async_op=True
    ).get_future()

    def unpack_uncompressed_tensors_and_allreduce_ps(fut):
        uncompressed_tensors_memory = fut.value()[0].div_(world_size)
        idx = 0
        for tensor in uncompressed_tensors:
            tensor.copy_(
                uncompressed_tensors_memory[idx : idx + tensor.numel()].view_as(tensor)
            )
            idx += tensor.numel()

        # Since these Ps will be orthogonalized later, no need to divide them by world size.
        return [
            dist.all_reduce(
                state.p_memory_dict[bucket_index], group=group_to_use, async_op=True
            )
            .get_future()
            .wait()[0]
        ]

    def compute_qs(fut):
        state.p_memory_dict[bucket_index] = fut.value()[0].div_(world_size)

        # # Compute Qs.
        # for tensor, p, q in zip(tensors_to_compress, ps, qs):
        #     torch.matmul(tensor.t(), p, out=q)

        # TODO: The above procedure does two matmul+allreduce steps per iteration --
        # one left multiplication and one right multiplication.
        # For warm-start, can take one such step at a time, and alternate between them.

        # Allreduce Qs.
        return [
            dist.all_reduce(
                state.q_memory_dict[bucket_index], group=group_to_use, async_op=True
            )
            .get_future()
            .wait()[0]
        ]

    def decompress(fut):
        state.q_memory_dict[bucket_index] = fut.value()[0].div_(world_size)

        for p, q, tensor in zip(ps, qs, tensors_to_compress):
            torch.matmul(p, q.t(), out=tensor)

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

        state.p_memory_dict.clear()
        state.q_memory_dict.clear()

        state.maybe_increase_iter(bucket)

        return [input_tensor]

    return (
        allreduce_contiguous_uncompressed_tensors_fut.then(
            unpack_uncompressed_tensors_and_allreduce_ps
        )
        .then(compute_qs)
        .then(decompress)
    )