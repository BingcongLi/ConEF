from torch.nn.parallel import DistributedDataParallel as ddp
import torch.distributed.algorithms.ddp_comm_hooks as ddp_comm
import communication.RandomBlock_hook as RandomBlock
import communication.UnbiasedRandomBlock_hook as UnbiasedRandomBlock
import communication.powerSGD_hook as powerSGD
import communication.hPowerSGD_hook as hPowerSGD


def get_comm_hooks(args, net, gpu):
    """
    register communication hook for ddp
    see https://pytorch.org/docs/stable/ddp_comm_hooks.html

    if allreduce, return None
    if EF or ConEF, return a the error vector or compressed error
    """

    #support allreduce, ef_randomblock, conef_randomblock, unbiased_randomblock

    reducer_name = args.grad_reducer.lower()
    start_compression_iter = args.start_compression_iter

    # allreduce
    if reducer_name == 'allreduce':
        state = None

    # random-block-k based reducers
    elif reducer_name == 'ef_randomblock':
        compression_ratio = args.compression_ratio
        state = RandomBlock.RandomBlockState(
            process_group=None,
            compression_ratio=compression_ratio,
            start_compression_iter=start_compression_iter,
        )
        hook = RandomBlock.RandomBlock_hook
        net.register_comm_hook(state, hook)
    elif reducer_name == 'conef_randomblock':
        compression_ratio = args.compression_ratio
        beta = args.beta
        sketch_size = args.sketch_size
        state = RandomBlock.RandomBlockState(
            process_group=None,
            compression_ratio=compression_ratio,
            start_compression_iter=start_compression_iter,
            use_error_compression=True,
            sketch_size=sketch_size,
            beta=beta,
        )
        hook = RandomBlock.RandomBlock_hook
        net.register_comm_hook(state, hook)
    elif reducer_name == 'unbiased_randomblock':
        compression_ratio = args.compression_ratio
        state = UnbiasedRandomBlock.UnbiasedRandomBlockState(
            process_group=None,
            compression_ratio=compression_ratio,
            start_compression_iter=start_compression_iter,
        )
        hook = UnbiasedRandomBlock.UnbiasedRandomBlock_hook
        net.register_comm_hook(state, hook)


    # powersgd type reducers
    elif reducer_name == 'efsgd_powersgd':
        matrix_approximation_rank = args.matrix_approximation_rank
        start_compression_iter = args.start_compression_iter
        min_compression_rate = args.min_compression_rate # typically set to 2
        orthogonalization_epsilon = args.orthogonalization_epsilon
        state = powerSGD.PowerSGDState(
                process_group=None,
                matrix_approximation_rank=matrix_approximation_rank,
                start_powerSGD_iter=start_compression_iter,
                min_compression_rate=min_compression_rate,
                orthogonalization_epsilon=orthogonalization_epsilon
            )
        hook = powerSGD.powerSGD_hook
        net.register_comm_hook(state, hook)
    elif reducer_name == 'hefsgd_powersgd':
        matrix_approximation_rank = args.matrix_approximation_rank
        start_compression_iter = args.start_compression_iter
        min_compression_rate = args.min_compression_rate  # typically set to 2
        orthogonalization_epsilon = args.orthogonalization_epsilon
        state = hPowerSGD.hPowerSGDState(
            process_group=None,
            matrix_approximation_rank=matrix_approximation_rank,
            start_powerSGD_iter=start_compression_iter,
            min_compression_rate=min_compression_rate,
            orthogonalization_epsilon=orthogonalization_epsilon,
            use_error_compression=True,
            topk_ratio=0.4
        )
        hook = hPowerSGD.hpowerSGD_hook
        net.register_comm_hook(state, hook)
    elif reducer_name == 'conef_powersgd':
        matrix_approximation_rank = args.matrix_approximation_rank
        start_compression_iter = args.start_compression_iter
        min_compression_rate = args.min_compression_rate  # typically set to 2
        orthogonalization_epsilon = args.orthogonalization_epsilon
        beta = args.beta
        sketch_size = args.sketch_size
        state = powerSGD.PowerSGDState(
            process_group=None,
            matrix_approximation_rank=matrix_approximation_rank,
            start_powerSGD_iter=start_compression_iter,
            min_compression_rate=min_compression_rate,
            orthogonalization_epsilon=orthogonalization_epsilon,
            sketch_size=sketch_size,
            beta=beta,
            use_error_compression=True
        )
        hook = powerSGD.powerSGD_hook
        net.register_comm_hook(state, hook)
    else:
        raise ValueError("Communication hook unsupported error.")

    return state




# from torch.nn.parallel import DistributedDataParallel as ddp
# import torch.distributed.algorithms.ddp_comm_hooks as ddp_comm
# # import ddp_hooks as ddp_comm
# # import ddp_hooks.powerSGD_hook
# # from ddp_hooks import C2CS_hook
# import torch
# # import ddp_hooks.powerSGD_hook
#
# def get_comm_hooks(args, net, gpu):
#
#     reducer_name = args.grad_reducer
#     reducer_name = reducer_name.lower()
#     if args.nr * args.gpus + gpu == 0:
#         print('communication hook:' + reducer_name)
#
#     matrix_approximation_rank = args.matrix_approximation_rank
#     start_compression_iter = args.start_compression_iter
#     min_compression_rate = args.min_compression_rate # typically set to 2
#     orthogonalization_epsilon = args.orthogonalization_epsilon
#
#     if reducer_name == 'allreduce':
#         state = None
#     elif reducer_name == 'efsgd_powersgd':
#         state = ddp_comm.powerSGD_hook.PowerSGDState(
#             process_group=None,
#             matrix_approximation_rank=matrix_approximation_rank,
#             start_powerSGD_iter=start_compression_iter,
#             min_compression_rate=min_compression_rate,
#             orthogonalization_epsilon=orthogonalization_epsilon
#         )
#         hook = ddp_comm.powerSGD_hook.powerSGD_hook
#         net.register_comm_hook(state, hook)
#     elif reducer_name == 'hefsgd_powersgd':
#         state = ddp_comm.hPowerSGD_hook.hPowerSGDState(
#             process_group=None,
#             matrix_approximation_rank=matrix_approximation_rank,
#             start_powerSGD_iter=start_compression_iter,
#             min_compression_rate=min_compression_rate,
#             orthogonalization_epsilon=orthogonalization_epsilon,
#             use_error_compression=True,
#             topk_ratio=0.4
#         )
#         hook = ddp_comm.hPowerSGD_hook.hpowerSGD_hook
#         net.register_comm_hook(state, hook)
#     elif reducer_name == 'c2cs_powersgd':
#         beta = args.beta
#         sketch_size = args.sketch_size
#         state = ddp_comm.powerSGD_hook.PowerSGDState(
#             process_group=None,
#             matrix_approximation_rank=matrix_approximation_rank,
#             start_powerSGD_iter=start_compression_iter,
#             min_compression_rate=min_compression_rate,
#             orthogonalization_epsilon=orthogonalization_epsilon,
#             sketch_size=sketch_size,
#             beta=beta,
#             use_error_compression=True
#         )
#         hook = ddp_comm.powerSGD_hook.powerSGD_hook
#         net.register_comm_hook(state, hook)
#     elif reducer_name == 'c2cs_v2_powersgd':
#         beta = args.beta
#         sketch_size = args.sketch_size
#         state = ddp_comm.powerSGD_hook.PowerSGDState(
#             process_group=None,
#             matrix_approximation_rank=matrix_approximation_rank,
#             start_powerSGD_iter=start_compression_iter,
#             min_compression_rate=min_compression_rate,
#             orthogonalization_epsilon=orthogonalization_epsilon,
#             sketch_size=sketch_size,
#             beta=beta,
#             use_error_compression=True
#         )
#         hook = ddp_comm.powerSGD_hook.C2CS_hook_v2
#         net.register_comm_hook(state, hook)
#     elif reducer_name == 'unbiased_mtrx':
#         state = ddp_comm.unbiasedMtrx_hook.unbiasedMtrxState(
#             process_group=None,
#             matrix_approximation_rank=matrix_approximation_rank,
#             start_powerSGD_iter=start_compression_iter,
#             min_compression_rate=min_compression_rate,
#         )
#         hook = ddp_comm.unbiasedMtrx_hook.UnbiasedMtrx_hook
#         net.register_comm_hook(state, hook)
#     elif reducer_name == 'efsgd_randomk':
#         compression_ratio = args.compression_ratio
#         state = ddp_comm.RandomK_hook.RandomKState(
#             process_group=None,
#             compression_ratio=compression_ratio,
#             start_compression_iter=start_compression_iter
#         )
#         hook = ddp_comm.RandomK_hook.RandomK_hook
#         net.register_comm_hook(state, hook)
#     elif reducer_name == 'c2cs_randomk':
#         compression_ratio = args.compression_ratio
#         beta = args.beta
#         sketch_size = args.sketch_size
#         state = ddp_comm.RandomK_hook.RandomKState(
#             process_group=None,
#             compression_ratio=compression_ratio,
#             start_compression_iter=start_compression_iter,
#             use_error_compression=True,
#             sketch_size=sketch_size,
#             beta=beta,
#         )
#         hook = ddp_comm.RandomK_hook.RandomK_hook
#         net.register_comm_hook(state, hook)
#     elif reducer_name == 'efsgd_randomblock':
#         compression_ratio = args.compression_ratio
#         state = ddp_comm.RandomBlock_hook.RandomBlockState(
#             process_group=None,
#             compression_ratio=compression_ratio,
#             start_compression_iter=start_compression_iter,
#         )
#         hook = ddp_comm.RandomBlock_hook.RandomBlock_hook
#         net.register_comm_hook(state, hook)
#     elif reducer_name == 'c2cs_randomblock':
#         compression_ratio = args.compression_ratio
#         beta = args.beta
#         sketch_size = args.sketch_size
#         state = ddp_comm.RandomBlock_hook.RandomBlockState(
#             process_group=None,
#             compression_ratio=compression_ratio,
#             start_compression_iter=start_compression_iter,
#             use_error_compression=True,
#             sketch_size=sketch_size,
#             beta=beta,
#         )
#         hook = ddp_comm.RandomBlock_hook.RandomBlock_hook
#         net.register_comm_hook(state, hook)
#     elif reducer_name == 'unbiasedrandomblock':
#         compression_ratio = args.compression_ratio
#         state = ddp_comm.UnbiasedRandomBlock_hook.UnbiasedRandomBlockState(
#             process_group=None,
#             compression_ratio=compression_ratio,
#             start_compression_iter=start_compression_iter,
#         )
#         hook = ddp_comm.UnbiasedRandomBlock_hook.UnbiasedRandomBlock_hook
#         net.register_comm_hook(state, hook)
#     else:
#         raise ValueError("Communication hook not supported yet.")
#
#     return state