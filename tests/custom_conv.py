# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
cudnn.benchmark = False
cudnn.deterministic = True
torch.manual_seed(7)

def conv2d_forward(x, w, stride=1, padding=0, dilation=1):
    N = x.shape[0]
    Cin = x.shape[1]
    Cout = w.shape[0]
    input_size = (x.shape[2], x.shape[3])
    kernel_size = (w.shape[2], w.shape[3])
    height = (input_size[0] + 2*padding- dilation *(kernel_size[0]-1)-1)//stride + 1
    width = (input_size[1] + 2*padding- dilation *(kernel_size[1]-1)-1)//stride + 1
    output_shape = (N, Cout, width, height)
    o = torch.zeros(output_shape, device=x.device, dtype=x.dtype)
    for i in range(N):
        for jCout in range(Cout):
            for k in range(width):
                for l in range(height):
                    e = 0.0
                    for jCin in range(Cin):
                        for iKernel in range(kernel_size[0]):
                            for jKernel in range(kernel_size[1]):
                                e += x[i, jCin, k+iKernel, l+jKernel] * w[jCout, jCin, iKernel, jKernel]
                    o[i, jCout, k, l] = e
    return o

def naive_conv2d_backward_input(input_shape, w, grad, stride=1, padding=0, dilation=1):
    input_size = (input_shape[2], input_shape[3])
    kernel_size = (w.shape[2], w.shape[3])
    N = input_shape[0]
    Cin = input_shape[1]
    Cout = w.shape[0]
    Wout = grad.shape[2]
    Hout = grad.shape[3]
    dx = torch.zeros(input_shape, device=w.device, dtype=w.dtype)
    padding_w_output = (input_size[0]+kernel_size[0]-1-Wout)//2
    padding_h_output = (input_size[1]+kernel_size[1]-1-Hout)//2
    for i in range(N):
        for jCin in range(Cin):
            for k in range(input_size[0]):
                for l in range(input_size[1]):
                    e = 0.0
                    for jCout in range(Cout):
                        for iKernel in range(kernel_size[0]):
                            for jKernel in range(kernel_size[1]):
                                i_out = k+iKernel-padding_w_output
                                j_out = l+jKernel-padding_h_output
                                rot180_i = (kernel_size[0]-iKernel-1) % kernel_size[0]
                                rot180_j = (kernel_size[1]-jKernel-1) % kernel_size[1]
                                if j_out >= 0 and i_out >= 0 and i_out < Wout and j_out < Hout:
                                    #print('[%d,%d,%d,%d]=[%d, %d] * [%d, %d](%d,%d)' % (i, jCin, k, l, i_out, j_out, rot180_i, rot180_j, iKernel, jKernel))
                                    e += grad[i, jCout, i_out, j_out] * w[jCout, jCin, rot180_i, rot180_j]
                    dx[i, jCin, k, l] = e
    return dx

def naive_conv2d_backward_weight(input, weight_shape, grad):
    kernel_size = (weight_shape[2], weight_shape[3])
    N = input.shape[0]
    Cin = input.shape[1]
    Cout = weight_shape[0]
    Wout = grad.shape[2]
    Hout = grad.shape[3]
    dw = torch.zeros(weight_shape, device=grad.device, dtype=grad.dtype)
    for i in range(Cout):
        for j in range(Cin):
            for k in range(kernel_size[0]):
                for l in range(kernel_size[0]):
                    e = 0.0
                    for iN in range(N):
                        for iWout in range(Wout):
                            for jHout in range(Hout):
                                e += grad[iN, i, iWout, jHout] * input[iN, j, iWout+k, jHout+l]
                    dw[i, j, k, l] = e
    return dw

def conv2d_backward_input(input_shape, w, grad, stride=1, padding=0, dilation=1):
    input_size = (input_shape[2], input_shape[3])
    kernel_size = (w.shape[2], w.shape[3])
    N = input_shape[0]
    Cin = input_shape[1]
    Cout = w.shape[0]
    Wout = grad.shape[2]
    Hout = grad.shape[3]
    dx = torch.zeros(input_shape, device=w.device, dtype=w.dtype)
    padding_w_output = (input_size[0]+kernel_size[0]-1-Wout)//2
    padding_h_output = (input_size[1]+kernel_size[1]-1-Hout)//2
    for i in range(N):
        for jCin in range(Cin):
            for k in range(input_size[0]):
                for l in range(input_size[1]):
                    for iKernel in range(kernel_size[0]):
                        for jKernel in range(kernel_size[1]):
                            i_out = k+iKernel-padding_w_output
                            j_out = l+jKernel-padding_h_output
                            rot180_i = (kernel_size[0]-iKernel-1) % kernel_size[0]
                            rot180_j = (kernel_size[1]-jKernel-1) % kernel_size[1]
                            if j_out >= 0 and i_out >= 0 and i_out < Wout and j_out < Hout:
                                dx[i, jCin, k, l] += (grad[i, :, i_out, j_out] * w[:, jCin, rot180_i, rot180_j]).sum()
    return dx

def conv2d_backward_weight(input, weight_shape, grad):
    kernel_size = (weight_shape[2], weight_shape[3])
    N = input.shape[0]
    Cin = input.shape[1]
    Cout = weight_shape[0]
    Wout = grad.shape[2]
    Hout = grad.shape[3]
    dw = torch.zeros(weight_shape, device=grad.device, dtype=grad.dtype)
    for i in range(Cout):
        for j in range(Cin):
            for k in range(kernel_size[0]):
                for l in range(kernel_size[0]):
                    dw[i, j, k, l] += (grad[:, i, :, :] * input[:, j, k:k+Wout, l:l+Hout]).sum()
    return dw


def test_conv_forward():
    N=8 # batch size
    Cin = 16 # the number of input channels
    Win = 16; Hin = 16
    Cout = 8 # the number of output channels
    K = 3 # kernel size 

    input = torch.randn(N, Cin, Win, Hin, requires_grad=True).cuda()
    weight = torch.randn(Cout, Cin, K, K, requires_grad=True).cuda()
    print('input shape: ', input.shape)
    print('weights shape: ', weight.shape)
    true_output = F.conv2d(input, weight)
    print('true output shape: ', true_output.shape)
    output = conv2d_forward(input, weight)
    print('forward diff: ', (output-true_output).norm())

    grad_output = torch.randn(output.shape).cuda()
    true_grad_input = F.grad.conv2d_input(input.shape, weight, grad_output) 
    grad_input = conv2d_backward_input(input.shape, weight, grad_output)
    print('backward input diff: ', (true_grad_input-grad_input).norm())

    true_grad_weight = F.grad.conv2d_weight(input, weight.shape, grad_output)
    grad_weight = conv2d_backward_weight(input, weight.shape, grad_output)
    print('backward weight diff: ', (true_grad_weight-grad_weight).norm())


if __name__ == '__main__':
    test_conv_forward()
