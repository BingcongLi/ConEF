import logging
import math

import numpy as np
import torch
import torch.distributed as dist

from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as default

# from . import default_hooks as default


def _orthogonalize(matrix, epsilon=0):
    """
    Applies Gram-Schmidt procedure to orthogonalize a given 2D tensor.
    If epsilon is 0, this is equivalent to `torch.qr(matrix, out=(matrix, _))`,
    """
    # TODO Consider using Q = torch.orgqr(*torch.geqrf(A)) to compute the Q of the QR _much_ faster
    # and more reliably.
    # Works on FP32/64 or complex numbers (does not work for half precision)
    num_cols = matrix.shape[1]
    for i in range(num_cols):
        # Normalize the i'th column.
        col = matrix[:, i : i + 1]
        # If no epsilon is added here, division by zero may be caused by vanishing gradients.
        # This epsilon is not needed if the input matrix covers the gradients of at least one entire layer in the neural network.
        if epsilon == 0:
            # Note that col ** 2 can underflow/overflow if we use FP16.
            # May need to consider multiplying a scaling factor and dividing it later, or using bfloat16 instead.
            try:
                col /= torch.norm(col)
            except ZeroDivisionError:
                logging.error(
                    "The matrix to be orthogonalized has at least a column of all 0s. Please set a small value such as 1e-8 "
                    "as `orthogonalization_epsilon` in PowerSGD state."
                )
                # Recover the values from NaNs to 0s.
                col.fill_(0.0)
        else:
            col /= torch.norm(col) + epsilon
        # Project it on the rest and remove it.
        if i + 1 < num_cols:
            rest = matrix[:, i + 1 :]
            rest -= torch.sum(col * rest, dim=0) * col


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


def _report_compression_stats(bucket, state):
    """
    Report compression stats at the frequency of `compression_stats_logging_frequency` specified in PowerSGD state.
    """
    if (
        bucket.is_the_last_bucket_to_allreduce()
        and state.iter >= state.next_stats_report
    ):
        stats = state.compression_stats()
        logging.info(
            "Compression stats: iter {}, total before compression {}, total after compression {}, "
            "rate {}".format(state.iter, stats[1], stats[2], stats[0])
        )
        state.next_stats_report = state.iter + state.compression_stats_logging_frequency


def _topk(error_tensor, topk_ratio):
    # here instead of stroing both indicies and values
    # we use a lazy manner just for simulation purpose
    top_size = max(1, int(0.5 * topk_ratio * error_tensor.nelement()))
    _, positions = torch.topk(error_tensor.view(-1).abs(), top_size, sorted=False)
    values = error_tensor.view(-1)[positions].contiguous()
    error_tensor.zero_()
    error_tensor.view(-1)[positions.long()] = values



class hPowerSGDState(object):
    r"""
    Stores both the algorithm's hyperparameters and the internal state for all the gradients during the training.
    Particularly, ``matrix_approximation_rank`` and ``start_powerSGD_iter`` are the main hyperparameters that should be tuned by the user.
    For performance, we suggest to keep binary hyperparameters ``use_error_feedback`` and ``warm_start`` on.

    1. ``matrix_approximation_rank`` controls the size of compressed low-rank tensors, which determines the compression rate. The lower the rank, the stronger the compression.

        1.1. If ``matrix_approximation_rank`` is too low, the full model quality will need more training steps to reach or will never reach and yield loss in accuracy.

        1.2. The increase of ``matrix_approximation_rank`` can substantially increase the computation costs of the compression, and the accuracy may not be futher improved beyond a certain ``matrix_approximation_rank`` threshold.

    To tune ``matrix_approximation_rank``, we suggest to start from 1 and increase by factors of 2 (like an expoential grid search, 1, 2, 4, ...), until a satisfactory accuracy is reached. Typically only a small value 1-4 is used. For some NLP tasks (as shown in Appendix D of the original paper), this value has been increased to 32.

    2. ``start_powerSGD_iter`` defers PowerSGD compression until step ``start_powerSGD_iter``, and vanilla allreduce runs prior to step ``start_powerSGD_iter``. This hybrid scheme of **vanilla allreduce + PowerSGD** can effectively improve the accuracy, even a relatively small ``matrix_approximation_rank`` is used. This is because that, the beginning of training phase is usually very sensitive to inaccurate gradients, and compressing gradients too early may make the training quickly take a suboptimal trajectory, which can result in an irrecoverable impact on the accuracy.

    To tune ``start_powerSGD_iter``, we suggest to start with 10% of total training steps, and increase it until a satisfactory accuracy is reached. If there is a warm-up stage in the training, ``start_powerSGD_iter`` typically should be no less than the number of warm-up steps.

    3. ``min_compression_rate`` is the minimum compression rate required when a layer is compressed. Due to the computation overheads incurred by the compression, a tensor is worth compressing only if there can be sufficient saving in bandwidth, where ``(num_rows + num_cols) * matrix_approximation_rank * min_compression_rate < num_rows * num_cols``. If the specified compression rate threshold cannot be satisfied, the tensor will be directly allreduced without compression.

    Compression statistics are logged every ``compression_stats_logging_frequency`` iterations once PowerSGD compression starts.

    4. ``orthogonalization_epsilon`` can be a very small value (e.g., 1e-8) added to every normalized matrix column in orthogonalization step, to prevent div-by-zero error if any column has all 0s. If this can already be prevented (e.g., by batch normalization), an epsilon of 0 is recommended for accuracy.

    .. warning ::
        If error feedback or warm-up is enabled, the minimum value of ``start_powerSGD_iter`` allowed in DDP is 2.
        This is because there is another internal optimization that rebuilds buckets at iteration 1 in DDP,
        and this can conflict with any tensor memorized before the rebuild process.
    """  # noqa: B950

    __slots__ = [
        "process_group",
        # The fields below are the hyperparameters that often need to be tuned by the user.
        "matrix_approximation_rank",
        "start_powerSGD_iter",
        # The fields below are the hyperparameters that seldom need be tuned by the user.
        "min_compression_rate",
        "orthogonalization_epsilon",
        "use_error_compression",
        "topk_ratio",
        # The fields below are the binary hyperparameters recommended to be turned on for performance and accuracy.
        "use_error_feedback",
        "warm_start",
        # The fields below are internal state.
        "rng",
        "error_dict",
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
        use_error_compression=False,
        use_error_feedback=True,
        topk_ratio=0.4,
        warm_start=True,
        orthogonalization_epsilon=0,
        random_seed=0,
        compression_stats_logging_frequency=10_000,
    ):

        self.process_group = process_group
        self.matrix_approximation_rank = matrix_approximation_rank
        # Deferring PowerSGD compression util step 'start_powerSGD_iter' can have two advantages:
        # 1) It turns out that PowerSGD may lead to a non-trivial accuracy loss,
        # even if the matrix approximation rank is increased to a large value.
        # To mitigate the accuracy loss, a simple yet effective way is mixing vanilla allreduce
        # (or a more conservative compression such as FP16 compression) with PowerSGD.
        # 2) There is an internal optimization of rebuilding buckets process in DDP,
        # in order to save the memory space.
        # This step takes place after the first iteration.
        # However, this means that the shape of input bucketized tensors is subject to change,
        # which will complicate the implementations of error feedback and warm-up.
        # Running vanilla allreduce in the first few iterations can avoid this complexity.
        if (use_error_feedback or warm_start) and start_powerSGD_iter <= 1:
            raise ValueError(
                "Expect `start_powerSGD_iter` > 1 if `use_error_feedback` or `warm_start` is enabled, "
                "because PowerSGD can only be applied after the first two iterations in DDP."
            )
        self.start_powerSGD_iter = start_powerSGD_iter
        self.min_compression_rate = min_compression_rate
        # Error feedback is usually crucial for both for convergence and generalization,
        # because PowerSGD is a biased compressor,
        # i.e., compressing and decompressing a random gradient does not yield the original in expectation.
        # This mechanism requires a temporary copy of the input gradients,
        # so it increases the peak memory consumption by the size of the gradient tensor.
        # However, if the target matrices are known to be exactly low-ranked (instead of just low stable rank),
        # sometimes it is possible to converge to the optima without error feedback.
        # See: http://proceedings.mlr.press/v54/yurtsever17a/yurtsever17a.pdf
        self.use_error_feedback = use_error_feedback
        # Warm-start reuses P(s) and Q(s) from the previous iteration.
        # This can improve the approximation quality and hence improve the accuracy.
        # Additionally, by avoiding the initialization of these low-rank tensors at every step,
        # this can also accelerate training.
        # However, this is at the cost of extra memory.
        self.warm_start = warm_start
        # Can use a very small value to prevent div-by-zero error caused by orthogonalization of vanishing gradients.
        self.orthogonalization_epsilon = orthogonalization_epsilon
        # The purpose of this RNG is to generate different random seeds for initializing Q across iterations,
        # but in the same order for all the DDP replicas.
        # Different random seeds across iterations indicate different 'projections' of the gradients at different SGD steps.
        # If the same random projection is used,
        # there will be differences between the gradients that are never synchronized.
        self.rng = np.random.RandomState(random_seed)
        # Since there is only a single state instance for all the input buckets,
        # need to maintain a dictionary that maps each bucket index to the local error.
        self.error_dict = {}
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
        # using top-k compressor for error vector
        self.use_error_compression = use_error_compression
        self.topk_ratio = topk_ratio

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

    def clean_error(self):
        if self.use_error_feedback:
            for key in self.error_dict.keys():
                for inner_key in self.error_dict[key].keys():
                    self.error_dict[key][inner_key].clean(0)


def hpowerSGD_hook(
    state: hPowerSGDState, bucket: dist.GradBucket
) -> torch.futures.Future:
    r"""
    This DDP communication hook implements PowerSGD gradient compression
    algorithm described in the `paper <https://arxiv.org/abs/1905.13727>`_.
    Once gradient tensors are aggregated across all workers, this hook applies
    compression as follows:

    1. Views the input flattened 1D gradient tensor as a list of per-parameter tensors, and divides all the tensors into two groups:

        1.1 The tensors that should be compressed before allreduce, because the compression can give enough saving in bandwidth.

        1.2 Rest of the tensors will be directly allreduced without compression, including all the vector tensors (for biases).

    2. Handles uncompressed tensors:

        2.1. Allocate contiguous memory for those uncompressed tensors, and allreduces all the uncompressed tensors as a batch, without compression;

        2.2. Copies the individual uncompressed tensors from the contiguous memory back to the input tensor.

    3. Handles the tensors that should be compressed by PowerSGD compression:

        3.1. For each tensor M, creates two low-rank tensors P and Q for decomposing M,
        such that M = PQ^T, where Q is initialized from a standard normal distribution and orthogonalized;

        3.2. Computes each P in Ps, which is equal to MQ;

        3.3. Allreduces Ps as a batch;

        3.4. Orthogonalizes each P in Ps;

        3.5. Computes each Q in Qs, which is approximately equal to M^TP;

        3.6. Allreduces Qs as a batch;

        3.7. Computes each M among all the compressed tensors, which is approximately equal to PQ^T.

    Note that this communication hook enforces vanilla allreduce for the first ``state.start_powerSGD_iter`` iterations.
    This not only gives the user more control over the tradeoff between speedup and accuracy,
    but also helps abstract away some complexity of the internal optimization of DDP for future communication hook developers.

    Args:
        state (PowerSGDState): State information to configure the compression rate and support error feedback, warm start, etc.
            To tune the compression configs, mainly need to tune ``matrix_approximation_rank``, ``start_powerSGD_iter``
            and ``min_compression_rate``.
        bucket (dist.GradBucket): Bucket that stores a 1D flattened gradient tensor that batches multiple per-variable tensors.
            Note that since DDP comm hook only supports single process single device mode at this time,
            only exactly one tensor is stored in this bucket.

    Returns:
        Future handler of the communication, which updates the gradients in place.

    """
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
    input_tensor_cp = None
    total_length = input_tensor.shape[0]
    if state.use_error_feedback and (not state.use_error_compression):
        if bucket_index in state.error_dict:
            input_tensor.add_(state.error_dict[bucket_index])
        else:
            logging.info(
                "A zero tensor of length {} that represents local error is created.".format(
                    total_length
                )
            )
            state.error_dict[bucket_index] = torch.zeros(
                total_length, device=device, dtype=dtype
            )

        # Keep a copy of the input tensor,
        # so that we can compute the local error caused by compression later,
        # by comparing this copy and the input tensor updated after decompression.
        input_tensor_cp = torch.clone(input_tensor).detach()

    if state.use_error_feedback and state.use_error_compression:
        if bucket_index in state.error_dict:
            pass
        else:
            state.error_dict[bucket_index] = {}
        # An empty list for copy necessary tensors
        copy_list = []

    # Unflatten the input tensor into per-parameter tensors, for layer-wise compression.
    tensors = bucket.get_per_parameter_tensors()

    # Step I: Divide all the tensors into two groups,
    # one will be compressed before allreduce and the other will be directly allreduced without compression.
    tensors_to_compress, uncompressed_tensors = [], []
    total_Ps_size = 0
    total_Qs_size = 0
    cnt = -1
    for tensor in tensors:
        matrix = tensor.view(tensor.shape[0], -1)
        n, m = matrix.shape
        matrix_approximation_rank = min(n, m, state.matrix_approximation_rank)
        compress_test = _should_compress(
            n, m, matrix_approximation_rank, state.min_compression_rate
        )
        state.total_numel_before_compression += compress_test[1]
        if compress_test[0]:
            cnt += 1
            if state.use_error_feedback and state.use_error_compression:
                # initialize count sketch
                if cnt not in state.error_dict[bucket_index]:
                    state.error_dict[bucket_index][cnt] = torch.zeros_like(matrix)
                # add compressed error back to matrix
                # state.error_dict[bucket_index][cnt].query(matrix, 1 - state.beta)
                matrix.add_(state.error_dict[bucket_index][cnt])
                copy_list.append(torch.clone(matrix).detach())

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
    need_randomize_qs = False
    if not state.warm_start or bucket_index not in state.p_memory_dict:
        need_randomize_qs = True
        # If warm-start is disabled, low-rank tensors will be initialized at every step.
        # Only log this if warm-start to avoid spamming.
        if state.warm_start:
            logging.info(
                "Allocating contiguous memory of length {} for Ps, and of length {} for Qs, respectively.".format(
                    total_Ps_size, total_Qs_size
                )
            )
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

    # If warm-start is enabled, reuse Qs from the previous iteration if possible and skip filling random values.
    # The exception is the first iteration when PowerSGD is applied.
    if not need_randomize_qs:
        for q in qs:
            _orthogonalize(q, state.orthogonalization_epsilon)
    else:
        with torch.random.fork_rng(devices=[]):
            # Fork this RNG to avoid changing the seed globally and affecting the random sampling anywhere else in the training.
            # The seed makes sure that the initial random values are the same across all the DDP replicas.
            # This seed should differ at every step.
            # Since it is very slow to fork RNG state across all the CUDA devices,
            # only fork on CPU and then move the generated tensor to the CUDA device (by overwriting q).
            torch.manual_seed(state.rng.randint(1_000_000_000))
            for q in qs:
                q.copy_(
                    torch.randn(
                        *q.shape,
                        device="cpu",
                        dtype=dtype,
                    )
                )
                _orthogonalize(q, state.orthogonalization_epsilon)

    # Compute Ps.
    for tensor, q, p in zip(tensors_to_compress, qs, ps):
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
        state.p_memory_dict[bucket_index] = fut.value()[0]
        for p in ps:
            _orthogonalize(p, state.orthogonalization_epsilon)

        # Compute Qs.
        for tensor, p, q in zip(tensors_to_compress, ps, qs):
            torch.matmul(tensor.t(), p, out=q)

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

        if state.use_error_feedback:
            # Memorize the local errors.
            if not state.use_error_compression:
                state.error_dict[bucket_index] = input_tensor_cp - input_tensor
            else:
                cnt = -1
                for tensor, cp_tensor in zip(tensors_to_compress, copy_list):
                    cnt += 1
                    state.error_dict[bucket_index][cnt] = cp_tensor - tensor
                    _topk(state.error_dict[bucket_index][cnt], state.topk_ratio)

        if not state.warm_start:
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
