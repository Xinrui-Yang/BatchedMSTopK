import copy
import torch
from torch import nn

from mpi4py import MPI


import numpy as np
import batched_tcmm
import time


import threading

def Batched_compress_tensor(tensor, ratio):
    with torch.no_grad():
        assert tensor.is_cuda

        col = tensor.size(1)
        # tensor = torch.nn.Flatten()(tensor) 
        k = max(int(col * ratio), 1)

        # print(type(k))

        start_time_cuda = time.perf_counter()
        value, index = batched_tcmm.f_batched_topk(tensor, k)
        end_time_cuda = time.perf_counter()
        total_time = end_time_cuda - start_time_cuda

        print(tensor)
        print(value)
        print(index)

        # index = index.to(dtype=torch.long)
        # values = tensor.data[index]
        # print(values)

        return total_time
    
def generate_tensor_sizes():
    """生成从1K到1G的所有tensor_size"""
    tensor_sizes = []
    size = 1024  # 1K
    while size <= 256 * 1024 * 1024:  # 256M
        tensor_sizes.append(size)
        size *= 2  # 生成1024, 2048, 4096, ..., 256M
    return tensor_sizes

def generate_levels():
    """返回level的取值"""
    return [4, 8, 16, 32, 64]

def generate_cols(tensor_sizes, level):
    """计算每个tensor_size对应的col值"""
    return [tensor_size // level for tensor_size in tensor_sizes]

def write_results_to_file(results, filename="BatchedMSTopK_results.txt"):
    """将处理结果写入文件"""
    with open(filename, 'w') as f:
        for result in results:
             f.write(f"batches={result[0]} tensor_size={result[1]} total_time={result[2]}\n")

class Data:
    def __init__(self, data):
        self.data = data

if __name__ == "__main__":
    process_id = 1
    device = torch.device(f"cuda:{process_id}")

    results = []
    levels = generate_levels()
    tensor_sizes = generate_tensor_sizes()

    for level in levels:
        print(level)
        cols = generate_cols(tensor_sizes, level)
        for col in cols:
            a = torch.rand([level, col]).to(device)
            data = Data(a)
            total_time = Batched_compress_tensor(data.data, 0.01)
            # print(level, col, total_time)
            results.append((level, col * level, total_time))

    write_results_to_file(results)
    print("Results written to 'results.txt'.")

    # a = torch.rand([4, 268435456]).to(device)
    # data = Data(a)
    # total_time = 0
    # total_time += Batched_compress_tensor(data.data, 0.01)
    # print("total_time: ",total_time)

    # num_iterations = 1
    # for _ in range(num_iterations):
    #     total_time_cuda += Batched_compress_tensor(data.data, 0.01)

    # average_time_cuda = total_time_cuda / num_iterations 
    # print("average_time_batched: ",average_time_cuda)
