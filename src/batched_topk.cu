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
#include "batched_topk.h"

#include <stdio.h>

// using namespace std;

struct StepInfo
{
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

// __global__ void SetStepInfo(float* temp, StepInfo* info, int level) {
//   info[0] = (StepInfo*)(temp + 1024);
//   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < level; i += gridDim.x * blockDim.x){
//     // info[i] = (StepInfo*)(info[0] + 1024 * i);
//     info[i] = (StepInfo*)(info[0] + i);
//   }
// }

// __global__ void InitStepInfo(StepInfo** info, int col, int level) {
//   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < level; i += gridDim.x * blockDim.x) {
//     info[i]->r = 1.0;
//     info[i]->l = 0.0;
//     info[i]->mid = 0.5;
//     info[i]->best_less_mid = 1.0;
//     info[i]->best_larger_mid = 0.0;
//     info[i]->best_less_count = 0;
//     info[i]->best_larger_count = col;
//     info[i]->total_larger_count = 0;
//     info[i]->begin_pos = 0;
//     info[i]->random_pos = 100000000;
//   }
//   // if (threadIdx.x == 0) printf("info[i]->best_larger_count = %d\n", info[0]->best_larger_count);
// }

__global__ void InitStepInfo(StepInfo* info, int col, int level) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < level; i += gridDim.x * blockDim.x) {
    info[i].r = 1.0;
    info[i].l = 0.0;
    info[i].mid = 0.5;
    info[i].best_less_mid = 1.0;
    info[i].best_larger_mid = 0.0;
    info[i].best_less_count = 0;
    info[i].best_larger_count = col;
    info[i].total_larger_count = 0;
    info[i].begin_pos = 0;
    info[i].random_pos = 100000000;
  }
  // if (threadIdx.x == 0) printf("info[i]->best_larger_count = %d\n", info[0]->best_larger_count);
}

__global__ void InitStepInfoMax_Min(StepInfo* info, float* tmp_max, float* tmp_min, int level) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < level; i += gridDim.x * blockDim.x) {
    info[i].max = tmp_max[i];
    info[i].min = tmp_min[i];

    // printf("info[%d].max = %f\n", i, info[i].max);
    // printf("info[%d].min = %f\n", i, info[i].min);
  }
}

int GetTopkComBufferSize(int block_size, int block_count, int random_times, int size, int level) {
  return 1024 * level * sizeof(int) + 1024 * level * sizeof(float) + level * sizeof(StepInfo);
}

__global__ void AddToLocalSum(int size, const float* input, float* local_sum) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {  
    float tmp = local_sum[i] + input[i];
    if (!isfinite(tmp)) tmp = 0.0;
    local_sum[i] = tmp;
  }
  // if (threadIdx.x == 0) printf("gridDim.x=%d\n", gridDim.x);
}

__global__ void MaxKernel(float *g_idata, float *g_odata, int size, int level, int grid) {
  // static shared memory
  __shared__ float smem[1024];
  int col = size/level;

  // set thread ID
  unsigned int tid = threadIdx.x;
  
  for(int k = 0; k < level; ++k){
    
    smem[tid] = 0;
    // __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x + k * col;
    if(i < (k + 1) * col){
      smem[tid] = fabs(g_idata[i]);

      for (i = i + blockIdx.x * blockDim.x; i < (k + 1) * col; i += gridDim.x * blockDim.x) {
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
      volatile float *vsmem = smem; 
      if (blockDim.x >= 64 && tid < 32)
      {
          if(vsmem[tid] < vsmem[tid + 32]) vsmem[tid] = vsmem[tid + 32];
          if(vsmem[tid] < vsmem[tid + 16]) vsmem[tid] = vsmem[tid + 16];
          if(vsmem[tid] < vsmem[tid + 8]) vsmem[tid] = vsmem[tid +  8];
          if(vsmem[tid] < vsmem[tid + 4]) vsmem[tid] = vsmem[tid +  4];
          if(vsmem[tid] < vsmem[tid + 2]) vsmem[tid] = vsmem[tid +  2];
          if(vsmem[tid] < vsmem[tid + 1]) vsmem[tid] = vsmem[tid +  1];
      }

      __syncthreads();

      // write result for this block to global mem
      if (tid == 0){ 
        g_odata[blockIdx.x + grid * k] = smem[0];    
        // printf("g_odata[%d]=%f\n",blockIdx.x + grid * k,smem[0]);         
        // if(grid == 1){
        //   printf("g_odata[%d]=%f\n",blockIdx.x + grid * k,smem[0]);        
        // }                        
      }
    }
  }
}

__global__ void MinKernel(float *g_idata, float *g_odata, int size, int level, int grid) {
  // static shared memory
  __shared__ float smem[1024];
  int col = size/level;

  // set thread ID
  unsigned int tid = threadIdx.x;

  for(int k = 0; k < level; ++k){
    smem[tid] = 0;
    int i = blockIdx.x * blockDim.x  + threadIdx.x + k * col;

    // __syncthreads();

    if(i < (k + 1) * col){
      // if(true){
      smem[tid] = fabs(g_idata[i]);

      // __syncthreads();

      for (i = i + blockIdx.x * blockDim.x; i < (k + 1) * col; i += gridDim.x * blockDim.x) {
        float tmp = fabs(g_idata[i]);
        if (tmp < smem[tid])
          smem[tid] = tmp;
      }

      __syncthreads();

      // in-place reduction and complete unroll
      if (blockDim.x >= 1024 && tid < 512) 
        if (smem[tid] > smem[tid + 512] && smem[tid + 512] > 0)
          smem[tid] = smem[tid + 512];
    
      __syncthreads();

      if (blockDim.x >= 512 && tid < 256)
        if (smem[tid] > smem[tid + 256] && smem[tid + 256] > 0)
            smem[tid] = smem[tid + 256];

      __syncthreads();

      if (blockDim.x >= 256 && tid < 128)  
        if (smem[tid] > smem[tid + 128] && smem[tid + 128] > 0)
          smem[tid] = smem[tid + 128];

      __syncthreads();

      if (blockDim.x >= 128 && tid < 64)  
        if (smem[tid] > smem[tid + 64] && smem[tid + 64] > 0)    
          smem[tid] = smem[tid + 64];

      __syncthreads();

      // if(tid < 10)
      //   printf("before: smem[%d] = %f\n", tid, smem[tid]);

      // unrolling warp
      
      if (blockDim.x >= 64 && tid < 32)
      {
        volatile float *vsmem = smem;
        if(vsmem[tid] > vsmem[tid + 32] && vsmem[tid + 32] > 0) vsmem[tid] = vsmem[tid + 32];
        if(vsmem[tid] > vsmem[tid + 16] && vsmem[tid + 16] > 0) vsmem[tid] = vsmem[tid + 16];
        if(vsmem[tid] > vsmem[tid + 8]&& vsmem[tid + 8] > 0) vsmem[tid] = vsmem[tid + 8];
        if(vsmem[tid] > vsmem[tid + 4]&& vsmem[tid + 4] > 0) vsmem[tid] = vsmem[tid + 4];
        if(vsmem[tid] > vsmem[tid + 2]&& vsmem[tid + 2] > 0) vsmem[tid] = vsmem[tid + 2];
        if(vsmem[tid] > vsmem[tid + 1]&& vsmem[tid + 1] > 0) vsmem[tid] = vsmem[tid + 1];
      }

      // write result for this block to global mem
      if (tid == 0) {
        g_odata[blockIdx.x + grid * k] = smem[0];
        // printf("g_odata[%d]=%f\n",blockIdx.x + grid * k,smem[0]);
        // if(size == 5) printf("g_odata[%d] = %f\n", blockIdx.x + grid * k, g_odata[blockIdx.x + grid * k]);
      }
    }
  }
}

__global__ void TopNumCudaKernel(StepInfo* info, const int size, const int kmax, float* in, int* larger_count, int level, int grid) {
  
  // static share memory
  __shared__ int temp_num[1024];  

  unsigned int tid = threadIdx.x;

  int col = size/level;

  for(int k = 0; k < level; ++k){
    if (info[k].total_larger_count == kmax) {
      continue;
    }

    float thd = (info[k].min) + info[k].mid * (info[k].max - info[k].min);
    // if(tid == 0) printf("thd = %f", thd);

    __syncthreads();

    temp_num[tid] = 0;  

    __syncthreads();

    for (int i = blockIdx.x * blockDim.x + threadIdx.x + k * col; i < (k + 1) * col; i += gridDim.x * blockDim.x) {
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
      volatile int *vsmem = temp_num;
      vsmem[tid] += vsmem[tid + 32];
      vsmem[tid] += vsmem[tid + 16];
      vsmem[tid] += vsmem[tid + 8]; 
      vsmem[tid] += vsmem[tid + 4]; 
      vsmem[tid] += vsmem[tid + 2]; 
      vsmem[tid] += vsmem[tid + 1];
    }
    if (tid == 0) {
      larger_count[blockIdx.x + grid * k] = temp_num[0];
      // printf("%d larger_count = %d\n", blockIdx.x, larger_count[blockIdx.x]);
    }
  }
}

__global__ void  SumNumCudaKernel(StepInfo* info, const int size, int* larger_count, const int kmax, int level, int grid) {

  // static shared memory
  __shared__ int smem[1024];

  unsigned int tid = threadIdx.x;
  int col = size/level;

  for(int k = 0; k < level; ++k){
    if (info[k].total_larger_count == kmax) {
      continue;
    }
    __syncthreads();

    smem[tid] = 0;
    
    __syncthreads();

    for (int i = blockIdx.x * blockDim.x + threadIdx.x + k * col; i < (k + 1) * col; i += gridDim.x * blockDim.x) {
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
      volatile int *vsmem = smem;
      vsmem[tid] += vsmem[tid + 32];
      vsmem[tid] += vsmem[tid + 16];
      vsmem[tid] += vsmem[tid + 8];
      vsmem[tid] += vsmem[tid + 4];
      vsmem[tid] += vsmem[tid + 2];
      vsmem[tid] += vsmem[tid + 1];
    }

    // write result for this block to global mem
    if (tid == 0){
      info[k].total_larger_count = smem[0];
      // printf("info[%d]->total_larger_count = %d\n", k, info[k]->total_larger_count);
    }
  }
}

__global__ void UpdateRatioKernel(StepInfo* info, const int kmax, int level) {
  for(int k = 0; k < level; ++k){
    if (info[k].total_larger_count < kmax && info[k].total_larger_count >= info[k].best_less_count) {
      info[k].best_less_mid = info[k].mid;
      info[k].best_less_count = info[k].total_larger_count;
    }
    if (info[k].total_larger_count > kmax && info[k].total_larger_count < info[k].best_larger_count) {
      info[k].best_larger_mid = info[k].mid;
      info[k].best_larger_count = info[k].total_larger_count;
    }

    //if (threadIdx.x == 0) printf("larger_count = %d, k = %d\n", info->total_larger_count, kmax);
    if (info[k].total_larger_count == kmax) {
      continue;
    }

    if (info[k].total_larger_count > kmax) {
      info[k].l = info[k].mid;
    } else {
      info[k].r = info[k].mid;
    }  
    info[k].mid = (info[k].l + info[k].r) / 2;
    // if (threadIdx.x == 0){
    //   printf("info[%d]->mid = %f\n", k, info[k]->mid);
    // }
  }
}

__global__ void SetRatioKernel1(StepInfo* info, const int kmax, int level) {
  //if (threadIdx.x == 0) printf("best less_count = %d, k = %d\n", info->best_less_count, kmax);
  for(int k = 0; k < level; ++k){
    if (info[k].total_larger_count == kmax) {
      continue;
    }

    info[k].mid = info[k].best_less_mid;
    info[k].total_larger_count = 0;
  }
}

__global__ void SetRatioKernel2(StepInfo* info, const int kmax, int level) {
  //printf("best larger_count = %d, k = %d, best_larget_mid = %f\n", info->best_larger_count, kmax, info->best_larger_mid);
  for(int k = 0; k < level; ++k){
    if (info[k].total_larger_count == kmax) {
      continue;
    }
    info[k].mid = info[k].best_larger_mid;
    info[k].begin_pos = info[k].total_larger_count;
    info[k].total_larger_count = 0;
  }
}

__global__ void TopkComCudaKernel(StepInfo* info, const int size, const int kmax, float* in, int* out, float* values, int* larger_count_perblock, int level, int grid) {

  unsigned int tid = threadIdx.x;

  __shared__ int smem[1024];

  int col = size/level;

  for(int k = 0; k < level; ++k){
    int num_begin = info[k].begin_pos;

    if (info[k].total_larger_count > kmax - info[k].begin_pos) {
      num_begin -= (info[k].random_pos % (info[k].total_larger_count - (kmax - info[k].begin_pos) + 1));
    }

    float thd = (info[k].min) + info[k].mid * (info[k].max - info[k].min);

    if (larger_count_perblock[blockIdx.x + k * grid] != 0) {
      for (int i = 0 + k * grid; i < blockIdx.x + k * grid; i++) {
        num_begin += larger_count_perblock[i];
      }

      smem[tid] = 0;
      for (int i = blockIdx.x * blockDim.x + threadIdx.x + k * col; i < (k + 1) * col; i += gridDim.x * blockDim.x) {
        smem[tid] += (fabs(in[i]) >= thd);
      }
      __syncthreads();

      for (int i = 0; i < tid; i++) 
        num_begin += smem[i]; //????????????????????????

      __syncthreads();
      
      for (int i = blockIdx.x * blockDim.x + threadIdx.x + k * col; i < (k + 1) * col; i += gridDim.x * blockDim.x) {
        if (fabs(in[i]) >= thd) {
          if (num_begin >= info[k].begin_pos && num_begin < kmax) {
            out[num_begin + k * kmax] = i - k * col;
            values[num_begin + k * kmax] = in[i];
            in[i] = 0;
          }
          num_begin++;
        }
      }
    }
  }
}

void TopkComFunctor(const cudaStream_t &s, int blocksize, int blockcount, int random_times, int level, 
                    int size, int col, int kmax, const float *in, float *local_sum, void *buffer, 
                    float *tmp_max, float *tmp_min, float *values, int *indices)
{
    blocksize = 1024;
    blockcount = 256;
    dim3 block(blocksize, 1);
    int blockcount2 = (col + blocksize - 1) / blocksize;
    if (blockcount2 < blockcount)
        blockcount = blockcount2;
    dim3 grid(blockcount, 1);

    AddToLocalSum<<<grid, block, 0, s>>>(size, in, local_sum);

    int* larger_count = (int*)buffer;
    float* temp = (float*)(larger_count + 1024 * level);
    StepInfo* info = (StepInfo*)(temp + 1024 * level);

    InitStepInfo<<<grid, block, 0, s>>>(info, col, level);

    MaxKernel<<<grid, block, 0, s>>>(local_sum, temp, size, level, grid.x);
    MaxKernel<<<1, block, 0, s>>>(temp, tmp_max, level * grid.x, level, 1);

    MinKernel<<<grid, block, 0, s>>>(local_sum, temp, size, level, grid.x);
    MinKernel<<<1, block, 0, s>>>(temp, tmp_min, level * grid.x, level, 1);

    InitStepInfoMax_Min<<<grid, block, 0, s>>>(info, tmp_max, tmp_min, level);

    for (int i = 0; i < random_times; ++i) {
      TopNumCudaKernel<<<grid, block, 0, s>>>(info, size, kmax, local_sum, larger_count, level, grid.x);
      SumNumCudaKernel<<<1, block, 0, s>>>(info, level * grid.x, larger_count, kmax, level, grid.x);
      UpdateRatioKernel<<<1, 1, 0, s>>>(info, kmax, level);
    }

    SetRatioKernel1<<<1, 1, 0, s>>>(info, kmax, level);
    TopNumCudaKernel<<<grid, block, 0, s>>>(info, size, kmax, local_sum, larger_count, level, grid.x);
    SumNumCudaKernel<<<1, block, 0, s>>>(info, level * grid.x, larger_count, kmax, level, grid.x);
    TopkComCudaKernel<<<grid, block, 0, s>>>(info, size, kmax, local_sum, indices, values, larger_count, level, grid.x);
    SetRatioKernel2<<<1, 1, 0, s>>>(info, kmax, level);
    TopNumCudaKernel<<<grid, block, 0, s>>>(info, size, kmax, local_sum, larger_count, level, grid.x);
    SumNumCudaKernel<<<1, block, 0, s>>>(info, level * grid.x, larger_count, kmax, level, grid.x);
    TopkComCudaKernel<<<grid, block, 0, s>>>(info, size, kmax, local_sum, indices, values, larger_count, level, grid.x);

}

std::vector<torch::Tensor> tcmm_batched_topk(torch::Tensor a, int k)
{
    auto a_shape = a.sizes();
    int level = a_shape[0];
    int col = a_shape[1];
    int a_count = a.numel();
    int n = a.numel();
    int block_size = 1024;
    int block_count = 256;
    int random_times = 30;
    int size = n * sizeof(float);
    const auto current_device = c10::cuda::current_device();
    c10::cuda::set_device(a.get_device());
    int sizetmp = GetTopkComBufferSize(block_size, block_count, random_times, n, level);
    cudaStream_t s;
    cudaStreamCreate(&s);
    float *input_d = a.data_ptr<float>();
    float *localsum_d, *tmp, *tmp_max, *tmp_min;
    cudaMalloc((void **)&tmp, sizetmp);
    cudaMalloc((void **)&localsum_d, size);

    cudaMalloc((void **)&tmp_max, level*sizeof(float));
    cudaMalloc((void **)&tmp_min, level*sizeof(float));
    
    auto options_int =
        torch::TensorOptions()
            .dtype(torch::kInt32)
            .layout(torch::kStrided)
            .device(a.device().type())
            .requires_grad(false);
    auto b = torch::zeros({level, k}, options_int);
    auto options_float =
        torch::TensorOptions()
            .dtype(a.dtype())
            .layout(torch::kStrided)
            .device(a.device().type())
            .requires_grad(false);
    auto c = torch::zeros({level, k}, options_float);
    int *index_d = b.data_ptr<int>();
    float *output_d = c.data_ptr<float>();
    TopkComFunctor(s, block_size, block_count, random_times, level, a_count, col, k, input_d, localsum_d, tmp, tmp_max, tmp_min, output_d, index_d);

    cudaStreamSynchronize(s);

    cudaFree(localsum_d);
    cudaFree(tmp);
    cudaFree(tmp_max);
    cudaFree(tmp_min);

    cudaStreamDestroy(s);

    std::vector<torch::Tensor> tuple;
    tuple.push_back(c);
    tuple.push_back(b);
    c10::cuda::set_device(current_device);
    return tuple;
}
