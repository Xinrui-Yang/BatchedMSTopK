# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import spdnn
torch.manual_seed(7)

a = torch.rand(6, 6).cuda()
a[a<0.6] = 0.0
at = a.t()
print('at: ', at)

b = torch.rand(6, 6).cuda()
print('b: ', b)
#c = spdnn.spmm(a, b)
print('at shape: ', at.shape)
torch.cuda.synchronize()
c = spdnn.sparse_t_x_dense(a, b)
print('c=axb: ', c)
c_true = at.mm(b)
print('c_true=axb: ', c_true)
print('norm: ', float((c-c_true).norm()))
