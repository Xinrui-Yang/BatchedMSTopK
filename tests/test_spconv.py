# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn.functional as F
import spdnn
torch.manual_seed(7)

input = torch.randn(2,3,3,3, requires_grad=True).cuda()
weight = torch.randn(3,3,2,2, requires_grad=True).cuda()
print('input shape: ', input.shape)
print('weights shape: ', weight.shape)

def test_conv():
    output = F.conv2d(input, weight)
    print('output shape: ', output.shape)
    return output

def test_conv_grad():
    grad_output = torch.randn(output.shape).cuda()
    print('grad_output: ', grad_output)
    grad_weight = F.grad.conv2d_weight(input, weight.shape, grad_output)
    print('grad_weight: ', grad_weight)
    grad_input = F.grad.conv2d_input(input.shape, weight, grad_output) 
    print('grad_weight: ', grad_input)

def conv2d(x, w, stride=1, padding=0, dilation=1):
    input_size = (x.shape[2], x.shape[3])
    kernel_size = (w.shape[2], w.shape[3])
    inp_unf = F.unfold(x, kernel_size)
    out_unf = w.view(w.size(0), -1).matmul(inp_unf)
    height = (input_size[0] + 2*padding- dilation *(kernel_size[0]-1)-1)//stride + 1
    width = (input_size[1] + 2*padding- dilation *(kernel_size[1]-1)-1)//stride + 1
    output_size = (height, width)
    output = out_unf.view(out_unf.shape[0], out_unf.shape[1], output_size[0], output_size[1])
    return output

def conv2d_grad_input(input_shape, w, grad, stride=1, padding=0, dilation=1):
    input_size = (input_shape[2], input_shape[3])
    kernel_size = (w.shape[2], w.shape[3])
    return conv2d(grad, w.tranpose(1,2))


def test_conv2gemm():
    kernel_size = (weight.shape[2], weight.shape[3])
    inp_unf = F.unfold(input, kernel_size)
    print('inp_unf shape: ', inp_unf.shape)
    #out_unf = inp_unf.transpose(1, 2).matmul(weight.view(weight.size(0), -1).t()).transpose(1, 2)
    out_unf = weight.view(weight.size(0), -1).mm(inp_unf)
    #w = weight.view(2, -1)
    #print('weight shape: ', weight.shape)
    #print('w shape: ', w.shape)
    #out_unf = w.mm(inp_unf)
    print('out_unf shape: ', out_unf.shape)
    #output = F.fold(out_unf, (2,2), kernel_size)
    output_size = (2, 2)
    output = out_unf.view(out_unf.shape[0], out_unf.shape[1], output_size[0], output_size[1])
    print('output shape: ', output.shape)
    return output


if __name__ == '__main__':
    output = test_conv()
    output2 = conv2d(input, weight)
    print('diff: ', (output-output2).norm())
