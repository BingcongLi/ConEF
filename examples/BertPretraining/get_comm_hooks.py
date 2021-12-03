from torch.nn.parallel import DistributedDataParallel as ddp
import torch.distributed.algorithms.ddp_comm_hooks as ddp_comm
import communication.RandomBlock_hook as RandomBlock
import communication.UnbiasedRandomBlock_hook as UnbiasedRandomBlock

def get_comm_hooks(args, net):
    # only support allreduce, ef_randomblock, conef_randomblock, unbiased_randomblock

    reducer_name = args.grad_reducer.lower()
    start_compression_iter = args.start_compression_iter

    if reducer_name == 'allreduce':
        state = None
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
    else:
        raise ValueError("Communication hook unsupported error.")

    return state