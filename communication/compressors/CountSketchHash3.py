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
	float* mem0,
	float* mem1,
	float* mem2,
	const float* beta_ptr,
	const int N,
	const int D,
	const int W)
{   
    const float beta = *beta_ptr;
	const int offset = blockIdx.x * D;
	int a[3] = {994443, 4113759, 9171025};
	int b[3] = {609478, 2949676, 2171464};
	float update[3];
	int hash_idx[3];
	float sign[3];

	// Read auxiliary variables
	extern __shared__ float shared[];
	float* aux0 = (float*) &shared[0];
	float* aux1 = (float*) &shared[W];
    float* aux2 = (float*) &shared[W+W];
	for(int index = threadIdx.x; index < W; index += blockDim.x)
	{   
	    // aux stores cs
		aux0[index] = mem0[blockIdx.x * W + index];
		aux1[index] = mem1[blockIdx.x * W + index];
		aux2[index] = mem2[blockIdx.x * W + index];
	}
	__syncthreads();

	// update sketch (e_t)
	for(int index = threadIdx.x; index < D; index += blockDim.x)
	{
		// Read vec
		float value = vec[offset + index];
		
		// Calculate auxiliary variable approximation 0
		hash_idx[0] = hash(index, W, a[0], b[0]);
		bool sign_bit = hash(index, W, a[0]+2, b[0]+3) & 0x1;
	    sign[0] = (sign_bit) ? 1.0 : -1.0;
	    update[0] = sign[0] * aux0[hash_idx[0]] * (beta - 1.0);
	    // update[0] = sign * (value - (1. - beta) * sign * aux0[hash_idx]);
		// atomicAdd(&mem0[blockIdx.x * W + hash_idx], update);
		// __syncthreads();
		
		// Calculate auxiliary variable approximation 1
		hash_idx[1] = hash(index, W, a[1], b[1]);
		sign_bit = hash(index, W, a[1]+4, b[1]+5) & 0x1;
	    sign[1] = (sign_bit) ? 1.0 : -1.0;
	    update[1] = (- (1. - beta) * sign[1] * aux1[hash_idx[1]]);
		// atomicAdd(&mem1[blockIdx.x * W + hash_idx], update);
		// __syncthreads();
		
		// Calculate auxiliary variable approximation 2
		hash_idx[2] = hash(index, W, a[2], b[2]);
		sign_bit = hash(index, W, a[2]+8, b[2]+4) & 0x1;
	    sign[2] = (sign_bit) ? 1.0 : -1.0;
	    update[2] = (- (1. - beta) * sign[2] * aux2[hash_idx[2]]);
	    
	    float update_fnl = median(update[0], update[1], update[2]);
	    
		atomicAdd(&mem0[blockIdx.x * W + hash_idx[0]], sign[0]*(update_fnl + value));
		atomicAdd(&mem1[blockIdx.x * W + hash_idx[1]], sign[1]*(update_fnl + value));
		atomicAdd(&mem2[blockIdx.x * W + hash_idx[2]], sign[2]*(update_fnl + value));
		
		__syncthreads();
		
	}
}


extern "C"
__global__
void dense_cs_query(float* parameter,
	float* mem0,
	float* mem1,
	float* mem2,
	const float* alpha_ptr,
	const int N,
	const int D,
	const int W)
{
	const float alpha = *alpha_ptr;
	const int offset = blockIdx.x * D;
	int a[3] = {994443, 4113759, 9171025};
	int b[3] = {609478, 2949676, 2171464};
	float update[3];

	// update parameters (p_t)
	for(int index = threadIdx.x; index < D; index += blockDim.x)
	{
		// Calculate auxiliary variable approximation
		// const int hash_idx = hash(index, W, a, b);
		// bool sign_bit = hash(index, W, a+2, b+3) & 0x1;
	    // float sign = (sign_bit) ? 1.0 : -1.0;
	    
	    int hash_idx = hash(index, W, a[0], b[0]);
		bool sign_bit = hash(index, W, a[0]+2, b[0]+3) & 0x1;
	    float sign = (sign_bit) ? 1.0 : -1.0;
	    update[0] = sign * mem0[blockIdx.x * W + hash_idx] * alpha;
	    
	    hash_idx = hash(index, W, a[1], b[1]);
		sign_bit = hash(index, W, a[1]+4, b[1]+5) & 0x1;
	    sign = (sign_bit) ? 1.0 : -1.0;
	    update[1] = sign * mem1[blockIdx.x * W + hash_idx] * alpha;
	    
	    hash_idx = hash(index, W, a[2], b[2]);
		sign_bit = hash(index, W, a[2]+8, b[2]+7) & 0x1;
	    sign = (sign_bit) ? 1.0 : -1.0;
	    update[2] = sign * mem2[blockIdx.x * W + hash_idx] * alpha;
	    
	    float update_fnl = median(update[0], update[1], update[2]);
		// float update = sign * mem[blockIdx.x * W + hash_idx] * alpha;
		// Perform parameter update
		atomicAdd(&parameter[offset + index], update_fnl);
		__syncthreads();
	}
}

'''


class DenseCSHash3:
    def __init__(self, N, D, sketch_size):
        self.N = N
        self.D = D
        self.blk_size = 512
        self.range = max(int(D * sketch_size), 1)
        device = torch.cuda.current_device()
        self.cs0 = torch.FloatTensor(self.N, self.range).fill_(0).to(device)
        self.cs1 = torch.FloatTensor(self.N, self.range).fill_(0).to(device)
        self.cs2 = torch.FloatTensor(self.N, self.range).fill_(0).to(device)
        self.kernel = None
        self.query_kernel = None
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
        state_dict['cs0'] = self.cs0.detach().cpu().numpy()
        state_dict['cs1'] = self.cs1.detach().cpu().numpy()
        state_dict['cs2'] = self.cs2.detach().cpu().numpy()
        return state_dict

    def __setstate__(self, d):
        self.__dict__ = d
        device = torch.cuda.current_device()
        self.cs0 = torch.from_numpy(self.cs0).to(device)
        self.cs1 = torch.from_numpy(self.cs1).to(device)
        self.cs2 = torch.from_numpy(self.cs2).to(device)
        self.kernel = None

    def initialize(self):
        if self.kernel is None:
            self.kernel = cupyKernel(kernel, "dense_cs_update")
            # print(1)

    def update(self, vec, beta):
        beta = torch.cuda.FloatTensor(1).fill_(beta)

        self.initialize()
        # shared memory - #copies x #elements x sizeof(float)
        self.kernel(grid=(self.N, 1, 1),
                    block=(self.blk_size, 1, 1),
                    args=[vec.data_ptr(),
                          self.cs0.data_ptr(),
                          self.cs1.data_ptr(),
                          self.cs2.data_ptr(),
                          beta.data_ptr(),
                          self.N,
                          self.D,
                          self.range],
                    strm=torch.cuda.current_stream().cuda_stream,
                    smem=int(16 * self.range))

    def query_initialize(self):
        if self.query_kernel is None:
            self.query_kernel = cupyKernel(kernel, "dense_cs_query")

    def query(self, grad_plus_error, alpha):
        self.query_initialize()
        alpha = torch.cuda.FloatTensor(1).fill_(alpha)
        # shared memory - #copies x #elements x sizeof(float)
        self.query_kernel(grid=(self.N, 1, 1),
                          block=(self.blk_size, 1, 1),
                          args=[grad_plus_error.data_ptr(),
                                self.cs0.data_ptr(),
                                self.cs1.data_ptr(),
                                self.cs2.data_ptr(),
                                alpha.data_ptr(),
                                self.N,
                                self.D,
                                self.range],
                          strm=torch.cuda.current_stream().cuda_stream,
                          smem=int(16 * self.range))

    def clean(self, alpha):
        self.cs0.mul_(alpha)
        self.cs1.mul_(alpha)
        self.cs2.mul_(alpha)


# printf("blockIdx.x is %d", blockIdx.x);