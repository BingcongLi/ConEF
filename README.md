#### ConEF
This repo contains code for contractive error feedback (ConEF) to reduce the memory overhead of EFSGD. 


ConEF is useful for training large models since it reduces the communication overhead while repsects to the memory constraint 
on GPUs. We expect ConEF to be useful when intergrating with ZeRO2 or ZeRO3. 
In particular, assuming we use a gradient compression that reduces 90% of 
communication overhead, and the memory-communication tradeoff between
different algorithms are summarized in the figure below.


#### Requirement:
- Python 3.7.10
- torch 1.9.0a0+gitd69c22d
- apex 0.1.0 ebcd7f0 (only for for BERT pretraining)

#### Code organization

The implementation of ConEF is based on torch.DistributedDataParallel. Main componets are
gradient and error compressors, which can be found at [here](https://github.com/BingcongLi/ConEF/tree/main/communication) and [here](https://github.com/BingcongLi/ConEF/tree/main/communication/compressors), respectively.

And here goes experiments on

- ResNet and WideResNet
- LSTM
- BERT pretraining
