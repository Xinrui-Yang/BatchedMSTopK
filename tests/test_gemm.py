# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import tcmm

n=10
k=4
a = torch.rand(n, k).cuda()
b = a.t()
refc = a @ b
print('a shape: ', a.shape)
print('b shape: ', b.shape)

print(a)
c = tcmm.f_gemm_ex(a, b)
print(a)
print((refc-c).norm())
#print(c)
