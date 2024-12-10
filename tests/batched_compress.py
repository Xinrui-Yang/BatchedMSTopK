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
        index = index.to(dtype=torch.long)
        # values = tensor.data[index]
        # print(values)

        return total_time

class Data:
    def __init__(self, data):
        self.data = data

if __name__ == "__main__":
    process_id = 0
    device = torch.device(f"cuda:{process_id}")
    a = torch.rand([32, 1000000]).to(device)
    data = Data(a)

    num_iterations = 1
    total_time_cuda = 0

    for _ in range(num_iterations):
        total_time_cuda += Batched_compress_tensor(data.data, 0.01)

    average_time_cuda = total_time_cuda / num_iterations 
    print("average_time_batched: ",average_time_cuda)
