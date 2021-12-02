A [ddp](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.) based implementation of ConEF.

In particular, the gradient compressors are implemented as communication hooks. To use them, one only needs to register one communication hook to ddp.


To enable error feedback (EF), one need to set
```
use_error_feedback = True
```
in the corresponding state for gradient compressors.

Similarly, ConEF is enabled by
```
use_error_feedback = True
use_error_compression = True
```

