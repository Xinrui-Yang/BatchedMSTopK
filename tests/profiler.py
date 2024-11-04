import torch.autograd.profiler as profiler
import torch
import tcmm
import time

from torch import nn

from mpi4py import MPI

import numpy as np


def compress_tensor(tensor, ratio):

    with torch.no_grad():
        assert tensor.is_cuda
        numel = tensor.numel()
        k = max(int(numel * ratio), 1)
        value, index = tcmm.f_topk(tensor, k)

class Data:
    def __init__(self, data):
        self.data = data        

def decorate_trace_handler(rank):
    def trace_handler(prof):
        if rank in [0]:
            prof.export_chrome_trace("mstopk_test"+str(rank)+".json")
    return trace_handler

prof = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=False,
        schedule=torch.profiler.schedule(
            wait=5,
            warmup=5,
            active=2),
        on_trace_ready=decorate_trace_handler(0)
    )

process_id = 0
device = torch.device(f"cuda:{process_id}")

with prof:
    for i in range(32):
        a = torch.rand([5, 5]).to(device)

        level = 5
        for i in range(level):
            data = Data(a[i])
            compress_tensor(data.data, 0.2)
        data = Data(a)
        prof.step()
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))