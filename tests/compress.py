import copy
import torch
from torch import nn

from mpi4py import MPI

import numpy as np
import tcmm
import time

import threading

def generate_tensor_sizes():
    """生成从1K到1G的所有tensor_size"""
    tensor_sizes = []
    size = 1024  # 1K
    while size <= 1024 * 1024 * 1024:  # 1G
        tensor_sizes.append(size)
        size *= 2  # 生成1024, 2048, 4096, ..., 1G
    return tensor_sizes

def generate_levels():
    """返回level的取值"""
    return [4, 8, 16, 32, 64]

def generate_cols(tensor_sizes, level):
    """计算每个tensor_size对应的col值"""
    return [tensor_size // level for tensor_size in tensor_sizes]

def compress_tensor(tensor, ratio):

    with torch.no_grad():
        assert tensor.is_cuda

        # tensor = torch.nn.Flatten()(tensor) 
        numel = tensor.numel()
        k = max(int(numel * ratio), 1)

        # print(type(k))

        start_time_cuda = time.perf_counter()
        value, index = tcmm.f_topk(tensor, k)
        end_time_cuda = time.perf_counter()
        total_time = end_time_cuda - start_time_cuda

        # print(tensor)
        # print(value)
        # print(index)
        return total_time
        # value = np.

        # value = value + 1

        # index = index.to(dtype=torch.long)
        # print(value)
        # print(index)

        # print(type(value))
        # print(type(index))

        # filed_value = tensor[index]
        # print(filed_value)


        # new_value = copy.deepcopy(value)
        # new_index = copy.deepcopy(index)

        # print(new_value)
        # print(new_index)

        # abs_tensor = torch.abs(tensor.data)
        # value, index = tcmm.f_topk(abs_tensor, k)
        # value = value + 1
        # value = value.cpu()
        # index = index.cpu()
        # print(value)

def write_results_to_file(results, filename="MSTopK_results.txt"):
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

    # col = int(tensor_size / level)

    for level in levels:
        print(level)
        cols = generate_cols(tensor_sizes, level)
        for col in cols:
            total_time = 0.0
            for i in range(level):
                a = torch.rand([col]).to(device)
                data = Data(a)
                MSTopK_time = compress_tensor(data.data, 0.01)    
                total_time += MSTopK_time
                # print(level, col, total_time)
            results.append((level, col * level, total_time))
    
    write_results_to_file(results)
    print("Results written to 'results.txt'.")
    
    # print("total_time:",total_time) 

    # num_iterations = 1
    # total_time_cuda = 0
    # total_time_1t = 0

    # for _ in range(num_iterations):
    #     level = 5
    #     for i in range(level):
    #         data = Data(a[i])
    #         total_time_1t += compress_tensor(data.data, 0.2)    
    # average_time_cuda = total_time_1t / num_iterations 
    # print("average_time_cuda:",average_time_cuda)

    

    # print("======================================================")
    # linear = nn.Linear(5, 5)
    # linear.to(device)
    # out = linear(a)
    # # out = copy.deepcopy(a)
    # out_data = Data(out)
    # new_data = Data(copy.deepcopy(out_data.data.data))

    # # new_compress_thread = threading.Thread(target=compress_tensor, args=(new_data.data, 0.2))
    # # new_compress_thread.start()
    # compress_tensor(data.data, 0.2)
