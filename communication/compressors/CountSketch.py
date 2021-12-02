import torch
from communication.compressors.cupy_kernel import cupyKernel
import numpy as np
import math

kernel = '''
extern "C"
__inline__ __device__
int hash(int value, int range, int a, int b)
{
	int h = a * value + b;
	h ^= h >> 16;
	h *= 0x85ebca6b;
	h ^= h >> 13;
	h *= 0xc2b2ae35;
	h ^= h >> 16;
	return h % range;
}

extern "C"
__inline__ __device__
float median(float a, float b, float c)
{
	return fmaxf(fminf(a,b), fminf(fmaxf(a,b),c));
}

extern "C"
__global__
void dense_cs_update(float* vec,
	float* mem,
	const float* beta_ptr,
	const int N,
	const int D,
	const int W,
	const int* a_ptr,
	const int* b_ptr)
{   
    const float beta = *beta_ptr;
	const int offset = blockIdx.x * D;
	const int a = *a_ptr;
	const int b = *b_ptr;

	// Read auxiliary variables
	extern __shared__ float shared[];
	float* aux = (float*) &shared[0];

	for(int index = threadIdx.x; index < W; index += blockDim.x)
	{   
	    // aux stores cs
		aux[index] = mem[blockIdx.x * W + index]; 
	}
	__syncthreads();

	// update sketch (e_t)
	for(int index = threadIdx.x; index < D; index += blockDim.x)
	{
		// Read vec
		float value = vec[offset + index];
		// Calculate auxiliary variable approximation
		int hash_idx = hash(index, W, a, b);
		bool sign_bit = hash(index, W, a+2, b+3) & 0x1;
	    float sign = (sign_bit) ? 1.0 : -1.0;
	    float update = sign * (value - (1. - beta) * sign * aux[hash_idx]);
		atomicAdd(&mem[blockIdx.x * W + hash_idx], update);
		__syncthreads();
	}
}


extern "C"
__global__
void dense_cs_query(float* parameter,
	float* mem,
	const float* alpha_ptr,
	const int N,
	const int D,
	const int W,
	const int* a_ptr,
	const int* b_ptr)
{
	const float alpha = *alpha_ptr;
	const int offset = blockIdx.x * D;
	const int a = *a_ptr;
	const int b = *b_ptr;

	// update parameters (p_t)
	for(int index = threadIdx.x; index < D; index += blockDim.x)
	{
		// Calculate auxiliary variable approximation
		int hash_idx = hash(index, W, a, b);
		bool sign_bit = hash(index, W, a+2, b+3) & 0x1;
	    float sign = (sign_bit) ? 1.0 : -1.0;
		float update = sign * mem[blockIdx.x * W + hash_idx] * alpha;
		// Perform parameter update
		atomicAdd(&parameter[offset + index], update);
		__syncthreads();
	}
}

'''

class DenseCS:
    def __init__(self, N, D, sketch_size):
        self.N = N
        self.D = D
        self.blk_size = 256
        self.range = max(int(D * sketch_size), 1)
        device = torch.cuda.current_device()
        self.cs = torch.FloatTensor(self.N, self.range).fill_(0).to(device)
        self.kernel = None
        self.query_kernel = None
        a = 994443
        b = 609478
        self.a = torch.cuda.IntTensor(1).fill_(a)
        self.b = torch.cuda.IntTensor(1).fill_(b)
        # print('N =', N, ' D =', D, 'range =', self.range)

    def state_dict(self):
        return self.__getstate__()

    def load_state_dict(self, d):
        return self.__setstate__(d)

    def __getstate__(self):
        state_dict = dict()
        state_dict['N'] = self.N
        state_dict['D'] = self.D
        state_dict['blk_size'] = self.blk_size
        state_dict['range'] = self.range
        state_dict['cs'] = self.cs.detach().cpu().numpy()
        return state_dict

    def __setstate__(self, d):
        self.__dict__ = d
        device = torch.cuda.current_device()
        self.cs = torch.from_numpy(self.cs).to(device)
        self.kernel = None

    def initialize(self):
        if self.kernel is None:
            self.kernel = cupyKernel(kernel, "dense_cs_update")

    def update(self, vec, beta):
        beta = torch.cuda.FloatTensor(1).fill_(beta)

        self.initialize()
        self.kernel(grid=(self.N, 1, 1),
                    block=(self.blk_size, 1, 1),
                    args=[vec.data_ptr(),
                          self.cs.data_ptr(),
                          beta.data_ptr(),
                          self.N,
                          self.D,
                          self.range,
                          self.a.data_ptr(),
                          self.b.data_ptr()],
                    strm=torch.cuda.current_stream().cuda_stream,
                    smem=int(8 * self.range))

    def query_initialize(self):
        if self.query_kernel is None:
            self.query_kernel = cupyKernel(kernel, "dense_cs_query")


    def query(self, grad_plus_error, alpha):
        self.query_initialize()
        alpha = torch.cuda.FloatTensor(1).fill_(alpha)
        self.query_kernel(grid=(self.N, 1, 1),
                          block=(self.blk_size, 1, 1),
                          args=[grad_plus_error.data_ptr(),
                                self.cs.data_ptr(),
                                alpha.data_ptr(),
                                self.N,
                                self.D,
                                self.range,
                                self.a.data_ptr(),
                                self.b.data_ptr()],
                          strm=torch.cuda.current_stream().cuda_stream,
                          smem=int(8 * self.range))

    def clean(self, alpha):
        self.cs.mul_(alpha)
