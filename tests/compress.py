import copy
import torch
from torch import nn

from mpi4py import MPI


import numpy as np
import tcmm
import time


import threading


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

        print(tensor)
        print(value)
        print(index)
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

class Data:
    def __init__(self, data):
        self.data = data


if __name__ == "__main__":
    process_id = 0
    device = torch.device(f"cuda:{process_id}")
    a = torch.rand([5]).to(device)
    data = Data(a)

    num_iterations = 1
    total_time_cuda = 0
    total_time_1t = 0

    # for _ in range(num_iterations):
    #     level = 5
    #     for i in range(level):
    #         data = Data(a[i])
    #         total_time_1t += compress_tensor(data.data, 0.2)    
    # average_time_cuda = total_time_1t / num_iterations 
    # print("average_time_cuda:",average_time_cuda)

    compress_tensor(data.data, 0.2)    

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
