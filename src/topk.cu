#include "cuda_runtime.h"
#include <cuda_fp16.h>
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <iostream>
#include <sys/time.h>
#include <algorithm>
#include <random>
#include <torch/extension.h>
#include <c10/cuda/CUDAFunctions.h>
#include "topk.h"

//using namespace std;

struct StepInfo {
    float r;
    float l;
    float max;
    float min;
    float mid;
    float best_less_mid;
    float best_larger_mid;
    int best_less_count;
    int best_larger_count;
    int total_larger_count;
    int begin_pos;
    int random_pos;
};

int GetTopkComBufferSize(int block_size, int block_count, int random_times, int size) {
  return 1024 * sizeof(int) + 1024 * sizeof(float) + sizeof(StepInfo);
}

__global__ void InitStepInfo(StepInfo* info, int size) {
  info->r = 1.0;
  info->l = 0.0;
  info->mid = 0.5;
  info->best_less_mid = 1.0;
  info->best_larger_mid = 0.0;
  info->best_less_count = 0;
  info->best_larger_count = size;
  info->total_larger_count = 0;
  info->begin_pos = 0;
  info->random_pos = 100000000;
}

__global__ void AddToLocalSum(int size, const float* input, float* local_sum) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    float tmp = local_sum[i] + input[i];
    if (!isfinite(tmp)) tmp = 0.0;
    local_sum[i] = tmp;
  }
}

__global__ void MaxKernel (float *g_idata, float *g_odata, int size) {
  // static shared memory
  __shared__ float smem[1024];

  // set thread ID
  unsigned int tid = threadIdx.x;

  int i = blockIdx.x * blockDim.x  + threadIdx.x;
  smem[tid] = fabs(g_idata[i]);
  for (i = i + blockIdx.x * blockDim.x; i < size; i += gridDim.x * blockDim.x) {
    float tmp = fabs(g_idata[i]);
    if (tmp > smem[tid])
      smem[tid] = tmp;
  }

  __syncthreads();

  // in-place reduction and complete unroll
  if (blockDim.x >= 1024 && tid < 512) 
    if (smem[tid] < smem[tid + 512])
      smem[tid] = smem[tid + 512];
 
  __syncthreads();

  if (blockDim.x >= 512 && tid < 256)
    if (smem[tid] < smem[tid + 256])
        smem[tid] = smem[tid + 256];

  __syncthreads();

  if (blockDim.x >= 256 && tid < 128)  
    if (smem[tid] < smem[tid + 128])
      smem[tid] = smem[tid + 128];

  __syncthreads();

  if (blockDim.x >= 128 && tid < 64)  
    if (smem[tid] < smem[tid + 64])    
      smem[tid] = smem[tid + 64];

  __syncthreads();

  // unrolling warp
  if (blockDim.x >= 64 && tid < 32)
  {
      volatile float *vsmem = smem;
      if(vsmem[tid] < vsmem[tid + 32]) vsmem[tid] = vsmem[tid + 32];
      if(vsmem[tid] < vsmem[tid + 16]) vsmem[tid] = vsmem[tid + 16];
      if(vsmem[tid] < vsmem[tid + 8]) vsmem[tid] = vsmem[tid +  8];
      if(vsmem[tid] < vsmem[tid + 4]) vsmem[tid] = vsmem[tid +  4];
      if(vsmem[tid] < vsmem[tid + 2]) vsmem[tid] = vsmem[tid +  2];
      if(vsmem[tid] < vsmem[tid + 1]) vsmem[tid] = vsmem[tid +  1];
  }

  // write result for this block to global mem
  if (tid == 0){
    g_odata[blockIdx.x] = smem[0];
    // printf("g_odata[%d]=%f\n",blockIdx.x,g_odata[blockIdx.x]);
  }
}

__global__ void MinKernel (float *g_idata, float *g_odata, int size) {
  // static shared memory
  __shared__ float smem[1024];

  // set thread ID
  unsigned int tid = threadIdx.x;

  int i = blockIdx.x * blockDim.x  + threadIdx.x;
  smem[tid] = fabs(g_idata[i]);
  for (i = i + blockIdx.x * blockDim.x; i < size; i += gridDim.x * blockDim.x) {
    float tmp = fabs(g_idata[i]);
    if (tmp < smem[tid])
      smem[tid] = tmp;
  }

  __syncthreads();

  // in-place reduction and complete unroll
  if (blockDim.x >= 1024 && tid < 512) 
    if (smem[tid] > smem[tid + 512])
      smem[tid] = smem[tid + 512];
 
  __syncthreads();

  if (blockDim.x >= 512 && tid < 256)
    if (smem[tid] > smem[tid + 256])
        smem[tid] = smem[tid + 256];

  __syncthreads();

  if (blockDim.x >= 256 && tid < 128)  
    if (smem[tid] > smem[tid + 128])
      smem[tid] = smem[tid + 128];

  __syncthreads();

  if (blockDim.x >= 128 && tid < 64)  
    if (smem[tid] > smem[tid + 64])    
      smem[tid] = smem[tid + 64];

  __syncthreads();

  // unrolling warp
  if (blockDim.x >= 64 && tid < 32)
  {
      volatile float *vsmem = smem;
      if(vsmem[tid] > vsmem[tid + 32]) vsmem[tid] = vsmem[tid + 32];
      if(vsmem[tid] > vsmem[tid + 16]) vsmem[tid] = vsmem[tid + 16];
      if(vsmem[tid] > vsmem[tid + 8]) vsmem[tid] = vsmem[tid +  8];
      if(vsmem[tid] > vsmem[tid + 4]) vsmem[tid] = vsmem[tid +  4];
      if(vsmem[tid] > vsmem[tid + 2]) vsmem[tid] = vsmem[tid +  2];
      if(vsmem[tid] > vsmem[tid + 1]) vsmem[tid] = vsmem[tid +  1];
  }

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = smem[0];
}

__global__ void UpdateRatioKernel(StepInfo* info, const int kmax) {
  if (info->total_larger_count < kmax && info->total_larger_count >= info->best_less_count) {
    info->best_less_mid = info->mid;
    info->best_less_count = info->total_larger_count;
  }
  if (info->total_larger_count > kmax && info->total_larger_count < info->best_larger_count) {
    info->best_larger_mid = info->mid;
    info->best_larger_count = info->total_larger_count;
  }

  //if (threadIdx.x == 0) printf("larger_count = %d, k = %d\n", info->total_larger_count, kmax);
  if (info->total_larger_count == kmax) {
    return;
  }

  if (info->total_larger_count > kmax) {
    info->l = info->mid;
  } else {
    info->r = info->mid;
  }  
  info->mid = (info->l + info->r) / 2;
}

__global__ void SetRatioKernel1(StepInfo* info, const int kmax) {
  //if (threadIdx.x == 0) printf("best less_count = %d, k = %d\n", info->best_less_count, kmax);
  if (info->total_larger_count == kmax) {
    return;
  }

  info->mid = info->best_less_mid;
  info->total_larger_count = 0;
}

__global__ void SetRatioKernel2(StepInfo* info, const int kmax) {
  //printf("best larger_count = %d, k = %d, best_larget_mid = %f\n", info->best_larger_count, kmax, info->best_larger_mid);

  if (info->total_larger_count == kmax) {
    return;
  }
  info->mid = info->best_larger_mid;
  info->begin_pos = info->total_larger_count;
  info->total_larger_count = 0;
}
__global__ void TopkComCudaKernel(StepInfo* info, const int size, const int kmax, float* in, int* out, float* values, int* larger_count_perblock) {

  unsigned int tid = threadIdx.x;
  int num_begin = info->begin_pos;
  if (info->total_larger_count > kmax - info->begin_pos) {
    num_begin -= (info->random_pos % (info->total_larger_count - (kmax - info->begin_pos) + 1));
  }

  float thd = (info->min) + info->mid * (info->max - info->min);


  if (larger_count_perblock[blockIdx.x] != 0) {
    for (int i = 0; i < blockIdx.x; i++) {
      num_begin += larger_count_perblock[i];
    }

    __shared__ int smem[1024];

    smem[tid] = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
      smem[tid] += (fabs(in[i]) >= thd);
    }
    __syncthreads();

    for (int i = 0; i < tid; i++) 
      num_begin += smem[i];

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
      if (fabs(in[i]) >= thd) {
        if (num_begin >= info->begin_pos && num_begin < kmax) {
          out[num_begin] = i;
          values[num_begin] = in[i];
          in[i] = 0;
        }
        num_begin++;
      }
    }
  }
}

__global__ void TopkComHalfCudaKernel(StepInfo* info, const int size, const int kmax, float* in, int* out, __half* values, int* larger_count_perblock) {
  unsigned int tid = threadIdx.x;
  int num_begin = info->begin_pos;
  if (info->total_larger_count > kmax - info->begin_pos) {
    num_begin -= (info->random_pos % (info->total_larger_count - (kmax - info->begin_pos) + 1));
  }

  float thd = (info->min) + info->mid * (info->max - info->min);


  if (larger_count_perblock[blockIdx.x] != 0) {
    for (int i = 0; i < blockIdx.x; i++) {
      num_begin += larger_count_perblock[i];
    }

    __shared__ int smem[1024];

    smem[tid] = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
      smem[tid] += (fabs(in[i]) >= thd);
    }
    __syncthreads();

    for (int i = 0; i < tid; i++) 
      num_begin += smem[i];

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
      if (fabs(in[i]) >= thd) {
        if (num_begin >= info->begin_pos && num_begin < kmax) {
          out[num_begin] = i;
          values[num_begin] = __float2half(in[i]);
          in[i] = 0;
        }
        num_begin++;
      }
    }
  }
}

__global__ void  SumNumCudaKernel(StepInfo* info, const int size, int* larger_count, const int kmax) {
  if (info->total_larger_count == kmax) {
    return;
  }
  
  // static shared memory
  __shared__ int smem[1024];

  unsigned int tid = threadIdx.x;

  smem[tid] = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    smem[tid] += larger_count[i];
  }
  __syncthreads();

  if (blockDim.x >= 1024 && tid < 512)
    smem[tid] += smem[tid + 512];

  __syncthreads();

  if (blockDim.x >= 512 && tid < 256)
    smem[tid] += smem[tid + 256];

  __syncthreads();

  if (blockDim.x >= 256 && tid < 128)
    smem[tid] += smem[tid + 128];

  __syncthreads();

  if (blockDim.x >= 128 && tid < 64)
    smem[tid] += smem[tid + 64];

  __syncthreads();

  if (blockDim.x >= 64 && tid < 32) {   
    volatile int* vsmem = smem;
    vsmem[tid] += vsmem[tid + 32];
    vsmem[tid] += vsmem[tid + 16];
    vsmem[tid] += vsmem[tid + 8];
    vsmem[tid] += vsmem[tid + 4];
    vsmem[tid] += vsmem[tid + 2];
    vsmem[tid] += vsmem[tid + 1];
  }

  // write result for this block to global mem
  if (tid == 0) info->total_larger_count = smem[0];
}

__global__ void TopNumCudaKernel(StepInfo* info, const int size, const int kmax, float* in, int* larger_count) {
  if (info->total_larger_count == kmax) {
    return;
  }

  // static share memory
  __shared__ int temp_num[1024];  

  unsigned int tid = threadIdx.x;
  float thd = (info->min) + info->mid * (info->max - info->min);
  //if (tid == 0 && blockIdx.x == 0) printf("r = %f, l = %f, mid = %f, min = %f, max = %f, thd = %f\n", info->r, info->l, info->mid, info->min, info->max, thd);

  temp_num[tid] = 0;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    temp_num[tid] += (fabs(in[i]) >= thd);
  }

  __syncthreads();

  if (blockDim.x >= 1024 && tid < 512)
    temp_num[tid] += temp_num[tid + 512];

  __syncthreads();

  if (blockDim.x >= 512 && tid < 256)
    temp_num[tid] += temp_num[tid + 256];

  __syncthreads();

  if (blockDim.x >= 256 && tid < 128)
    temp_num[tid] += temp_num[tid + 128];

  __syncthreads();

  if (blockDim.x >= 128 && tid < 64)
    temp_num[tid] += temp_num[tid + 64];

  __syncthreads();

  // unrolling warp
  // share memory
  if (blockDim.x >= 64 && tid < 32) {
    volatile int* vsmem = temp_num;
    vsmem[tid] += vsmem[tid + 32];
    vsmem[tid] += vsmem[tid + 16];
    vsmem[tid] += vsmem[tid + 8]; 
    vsmem[tid] += vsmem[tid + 4]; 
    vsmem[tid] += vsmem[tid + 2]; 
    vsmem[tid] += vsmem[tid + 1];
  }
  if (tid == 0) {
    larger_count[blockIdx.x] = temp_num[0];
    //printf("%d larger_count = %d\n", blockIdx.x, larger_count[blockIdx.x]);
  }
}

void TopkComFunctor(const cudaStream_t &s, int blocksize, int blockcount, int random_times,
  int size, int kmax, const float* in, float* local_sum, void* buffer, float* values, int* indices) {
  //size = a.size0

  blocksize = 1024;
  blockcount = 256;
  dim3 block(blocksize, 1);
  int blockcount2 = (size + blocksize - 1) / blocksize; 
  if (blockcount2 < blockcount) blockcount = blockcount2;
  dim3 grid(blockcount, 1);

  AddToLocalSum<<<grid, block, 0, s>>>(size, in, local_sum);

  int* larger_count = (int*)buffer;
  float* temp = (float*)(larger_count + 1024);
  StepInfo* info = (StepInfo*)(temp + 1024);

  InitStepInfo<<<1, 1, 0, s>>>(info, size);

  MaxKernel<<<grid, block, 0, s>>>(local_sum, temp, size);
  MaxKernel<<<1, block, 0, s>>>(temp, &info->max, grid.x);
  
  MinKernel<<<grid, block, 0, s>>>(local_sum, temp, size);
  MinKernel<<<1, block, 0, s>>>(temp, &info->min, grid.x);

  for (int i = 0; i < random_times; ++i) {
    TopNumCudaKernel<<<grid, block, 0, s>>>(info, size, kmax, local_sum, larger_count);
    SumNumCudaKernel<<<1, block, 0, s>>>(info, grid.x, larger_count, kmax);
    UpdateRatioKernel<<<1, 1, 0, s>>>(info, kmax);
  }

  SetRatioKernel1<<<1, 1, 0, s>>>(info, kmax);
  TopNumCudaKernel<<<grid, block, 0, s>>>(info, size, kmax, local_sum, larger_count);
  SumNumCudaKernel<<<1, block, 0, s>>>(info, grid.x, larger_count, kmax);
  TopkComCudaKernel<<<grid, block, 0, s>>>(info, size, kmax, local_sum, indices, values, larger_count);
  SetRatioKernel2<<<1, 1, 0, s>>>(info, kmax);
  TopNumCudaKernel<<<grid, block, 0, s>>>(info, size, kmax, local_sum, larger_count);
  SumNumCudaKernel<<<1, block, 0, s>>>(info, grid.x, larger_count, kmax);
  TopkComCudaKernel<<<grid, block, 0, s>>>(info, size, kmax, local_sum, indices, values, larger_count);
}

void TopkComHalfFunctor(int device_id, const cudaStream_t &s, int blocksize, int blockcount, int random_times,
  int size, int kmax, const float* in, float* local_sum, void* buffer, __half* values, int* indices) {
  cudaSetDevice(device_id);

  blocksize = 1024;
  blockcount = 256;
  dim3 block(blocksize, 1);
  int blockcount2 = (size + blocksize - 1) / blocksize;
  if (blockcount2 < blockcount) blockcount = blockcount2;
  dim3 grid(blockcount, 1);

  AddToLocalSum<<<grid, block, 0, s>>>(size, in, local_sum);

  int* larger_count = (int*)buffer;
  float* temp = (float*)(larger_count + 1024);
  StepInfo* info = (StepInfo*)(temp + 1024);

  InitStepInfo<<<1, 1, 0, s>>>(info, size);

  MaxKernel<<<grid, block, 0, s>>>(local_sum, temp, size);
  MaxKernel<<<1, block, 0, s>>>(temp, &info->max, grid.x);
  
  MinKernel<<<grid, block, 0, s>>>(local_sum, temp, size);
  MinKernel<<<1, block, 0, s>>>(temp, &info->min, grid.x);

  for (int i = 0; i < random_times; ++i) {
    TopNumCudaKernel<<<grid, block, 0, s>>>(info, size, kmax, local_sum, larger_count);  //统计大于阈值的元素的数量
    SumNumCudaKernel<<<1, block, 0, s>>>(info, grid.x, larger_count, kmax);
    UpdateRatioKernel<<<1, 1, 0, s>>>(info, kmax);
  }

  SetRatioKernel1<<<1, 1, 0, s>>>(info, kmax);
  TopNumCudaKernel<<<grid, block, 0, s>>>(info, size, kmax, local_sum, larger_count);
  SumNumCudaKernel<<<1, block, 0, s>>>(info, grid.x, larger_count, kmax);
  TopkComHalfCudaKernel<<<grid, block, 0, s>>>(info, size, kmax, local_sum, indices, values, larger_count);
  SetRatioKernel2<<<1, 1, 0, s>>>(info, kmax);
  TopNumCudaKernel<<<grid, block, 0, s>>>(info, size, kmax, local_sum, larger_count);
  SumNumCudaKernel<<<1, block, 0, s>>>(info, grid.x, larger_count, kmax);
  TopkComHalfCudaKernel<<<grid, block, 0, s>>>(info, size, kmax, local_sum, indices, values, larger_count);
}

__global__ void SumIndexKernel(int k, int n, const float* value, int* index, float* output, int* output_index) {
  int pos;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < k; i += gridDim.x * blockDim.x) {
    pos = index[i];
    if (pos < n) {
      output[pos] += value[i];
    }
    if (output_index != NULL) {
      output_index[pos] = 1;
    }
  }
}

__global__ void SumIndexHalfKernel(int k, int n, const __half* value, int* index, float* output, int* output_index) {
  int pos;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < k; i += gridDim.x * blockDim.x) {
    pos = index[i];
    if (pos < n) {
      output[pos] += __half2float(value[i]);
    }
    if (output_index != NULL) {
      output_index[pos] = 1;
    }
  }
}

void SumIndexFunctor(int device_id, cudaStream_t &s, int block_size, int block_count, int k, int times, int n, const float* value, int* index, float* output, int* output_index) {
  cudaSetDevice(device_id);
  cudaMemsetAsync(output, 0, n * sizeof(float), s);
  cudaMemsetAsync(output_index, 0, n * sizeof(int), s);
  for (int i = 0; i < times; ++i) {
    SumIndexKernel<<<block_count, block_size, 0, s>>>(k, n, value, index, output, output_index);
    value = (float*)((uint8_t*)value + k * (sizeof(float) + sizeof(int)));
    index = (int*)((uint8_t*)index + k * (sizeof(float) + sizeof(int)));
  }
}

void SumIndexHalfFunctor(int device_id, cudaStream_t &s, int block_size, int block_count, int k, int times, int n, const __half* value, int* index, float* output, int* output_index) {
  cudaSetDevice(device_id);
  cudaMemsetAsync(output, 0, n * sizeof(__half), s);
  cudaMemsetAsync(output_index, 0, n * sizeof(int), s);
  for (int i = 0; i < times; ++i) {
    SumIndexHalfKernel<<<block_count, block_size, 0, s>>>(k, n, value, index, output, output_index);
    value = (__half*)((uint8_t*)value + k * (sizeof(__half) + sizeof(int)));
    index = (int*)((uint8_t*)index + k * (sizeof(__half) + sizeof(int)));
  }
}

std::vector<torch::Tensor> tcmm_topk(torch::Tensor a, int k) {
    auto a_shape = a.sizes();
    int n = a_shape[0];
    int block_size = 1024;
    int block_count = 256;
    int random_times = 30;
    int size = n * sizeof(float);
    const auto current_device = c10::cuda::current_device();
    c10::cuda::set_device(a.get_device());
    int sizetmp = GetTopkComBufferSize(block_size, block_count, random_times, n);
    cudaStream_t s;
    cudaStreamCreate(&s);
    float *input_d = a.data_ptr<float>();
    float *localsum_d, *tmp;
    cudaMalloc((void**)&localsum_d, size);
    cudaMalloc((void**)&tmp, sizetmp);
    auto options_int =
        torch::TensorOptions()
        .dtype(torch::kInt32)
        .layout(torch::kStrided)
        .device(a.device().type())
        .requires_grad(false);
    auto b = torch::zeros({k}, options_int);
    auto options_float =
        torch::TensorOptions()
        .dtype(a.dtype())
        .layout(torch::kStrided)
        .device(a.device().type())
        .requires_grad(false); 
    auto c = torch::zeros({k}, options_float);
    int *index_d = b.data_ptr<int>();
    float *output_d = c.data_ptr<float>();
    TopkComFunctor(s, block_size, block_count, random_times, n, k, input_d, localsum_d, tmp, output_d, index_d);
    cudaStreamSynchronize(s);

    cudaFree(localsum_d);
    cudaFree(tmp);
    cudaStreamDestroy(s);
    std::vector<torch::Tensor> tuple;
    tuple.push_back(c);
    tuple.push_back(b);
    c10::cuda::set_device(current_device);
    return tuple;
}

