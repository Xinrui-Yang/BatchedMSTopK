# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import spdnn
import time
torch.manual_seed(7)

ITER=1000

M=2000
N=2000
K=2000
FLOP=M*N*K*2
GFLOP=FLOP/(1024*1024*1024)

a = torch.rand(K, M).cuda()
a[a<0.95] = 0.0
at = a.t()

b = torch.rand(K, N).cuda()
torch.cuda.synchronize()

# warmup
for i in range(5):
    c = at.mm(b)
    c = spdnn.sparse_t_x_dense(a, b)
torch.cuda.synchronize()

def bench_func(func):
    st = time.time()
    for i in range(ITER):
        func()
    torch.cuda.synchronize()
    et = time.time()
    avg_time = (et-st)/ITER
    return avg_time

def _func1():
    c = at.mm(b)
def _func2():
    c = spdnn.sparse_t_x_dense(a, b)
func1_time = bench_func(_func1)
func2_time = bench_func(_func2)
print('Dense time: %f sec, Perf: %f GFLOPS' % (func1_time, GFLOP/func1_time))
print('Sparse time: %f sec, Perf: %f GFLOPS' % (func2_time, GFLOP/func2_time))

