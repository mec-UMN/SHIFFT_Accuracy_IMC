import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import argparse
import random
import pandas as pd

def uniform_quantize(k, input):
    if k == 32:
        out = input
    elif k == 0:
         out = torch.sign(input)
        #import pdb;pdb.set_trace()
    else:
        n = float(2 ** k - 1)
        #if k == 1:
            #import pdb;pdb.set_trace()
        out = torch.round(input* n) / n
    return out

def uniform_quantize_bit_slice(k, input, width, E):
    if k == 32:
        out = input
    elif k == 1:
        out = torch.sign(input)
        #import pdb;pdb.set_trace()
        E_out = E
    else:
        n = int(2 ** (width-1) - 1)
        #import pdb;pdb.set_trace()
        indices = math.ceil((width)/(k))
        m=(math.pow(2, k)-1)
        quantize = torch.round(input* n)
        data, index= quantize.unique().abs().unique().sort()
        quantize_rp = quantize.repeat_interleave(indices, dim =1)
        out = torch.zeros_like(quantize_rp)
        for i in range(0,quantize_rp.size()[1]):
            if i%indices != indices-1:
                #import pdb;pdb.set_trace()
                out[:,i] = (quantize_rp[:,i]//(m+1))
                #quantize_rp[:,i+1] = torch.fmod(quantize_rp[:,i],(m+1))
            else:
                #out[:,i] = quantize_rp[:,i]
                out[:,i]=quantize_rp[:,i]-out[:,i-1]*(m+1)
        #print(out)
        #import pdb;pdb.set_trace()
        size = int(input.shape[0])
        index = torch.range(0,size-1,dtype=int)*2
        #import pdb;pdb.set_trace()
        output=torch.cat((out[:,index], out[:,index+1]),1)
        E_out = E/n
        #import pdb;pdb.set_trace()
    return output, E_out

def uniform_quantize_bit_slice_11(k, input, width, E):
    if k == 32:
        out = input
    elif k == 1:
        out = torch.sign(input)
        #import pdb;pdb.set_trace()
        E_out = E
    else:
        #n = int(2 ** (width-1) - 1)
        m=10
        n = m*(m+1)+m+1
        #import pdb;pdb.set_trace()
        indices = math.ceil((width)/(k))
        #m=(math.pow(2, k)-1)
        quantize = torch.round(input* n)
        data, index= quantize.unique().abs().unique().sort()
        quantize_rp = quantize.repeat_interleave(indices, dim =1)
        out = torch.zeros_like(quantize_rp)
        for i in range(0,quantize_rp.size()[1]):
            if i%indices != indices-1:
                #import pdb;pdb.set_trace()
                out[:,i] = (quantize_rp[:,i].abs()//(m+1))*torch.sign(quantize_rp[:,i])
                quantize_rp[:,i+1] = torch.fmod(quantize_rp[:,i],(m+1))
            else:
                out[:,i] = quantize_rp[:,i]
        #print(out)
        #import pdb;pdb.set_trace()
        out=out
        E_out = E/n
        #import pdb;pdb.set_trace()
    return out, E_out

def quantize_fn_uniform(k, input, E, alpha):
    if k == 32:
        out = input
        E=1
    else:
        if E is None:
            if alpha is None:
                E = torch.max(torch.abs(input)).detach()
                alpha = E/torch.max(torch.abs(input)).detach()
                if E ==0:
                    E =1
                    #import pdb;pdb.set_trace()
            else:
                E= alpha*torch.max(torch.abs(input)).detach()
        else:
            alpha = E/torch.max(torch.abs(input)).detach()
        x = torch.clamp(input, min = -E, max = E)
        input_norm = x/E
        #quantize = uniform_quantize(k-1, input_norm)
        quantize = uniform_quantize(k-1, input_norm)
       # import pdb;pdb.set_trace()
        out = (quantize)
    return out, E, alpha

class quantize_fn(torch.autograd.Function):
    def __init__(self,k):
		#import pdb;pdb.set_trace()
        super(quantize_fn, self).__init__()
        self.k = k

    def forward(self, input, E, alpha, signed):
        if self.k == 32:
            out = input
        else:
            #import pdb;pdb.set_trace()
            if E is None:
                if alpha is None:
                    E = torch.max(torch.abs(input)).detach()
                    alpha = E/torch.max(torch.abs(input)).detach()
                    #E = torch.max(torch.abs(input)).detach()
                    if E ==0:
                        E =1
                        #import pdb;pdb.set_trace()
                else:
                    E= alpha*torch.max(torch.abs(input)).detach()
            else:
                alpha = E/torch.max(torch.abs(input)).detach()
            self.save_for_backward(input, E)
            x = torch.clamp(input, min = -E, max = E)
            input_norm = x/E
            if signed:
                quant_level=self.k-1
            else:
                quant_level=self.k
            quantize = uniform_quantize(quant_level, input_norm)
            #import pdb;pdb.set_trace()
            out = (quantize)
        return out, E, alpha
    
    def backward(self, dLdy_q):
		# Backward function, I borrowed code from
		# https://github.com/obilaniu/GradOverride/blob/master/functional.py
		# We get dL / dy_q as a gradient
        x, E = self.saved_tensors
		# Weight gradient is only valid when [0, alpha]
		# Actual gradient for alpha,
		# By applying Chain Rule, we get dL / dy_q * dy_q / dy * dy / dalpha
		# dL / dy_q = argument,  dy_q / dy * dy / dalpha = 0, 1 with x value range 
        lower_bound      = x < -E
        upper_bound      = x > E
		# x_range       = 1.0-lower_bound-upper_bound
        x_range = ~(lower_bound|upper_bound)
        grad_alpha = torch.sum(dLdy_q * torch.ge(x, E).float()).view(-1)
        return dLdy_q * x_range.float(), grad_alpha, None

def quantize_fn_percentile(k, input, E, alpha):
    if k == 32:
        out = input
    else:
        #import pdb;pdb.set_trace()
        if E is None:
            if alpha is None:
                quantile=1-20/input.shape[0]
                E = input.abs().quantile(quantile, interpolation ='linear')
                #E = torch.max(torch.abs(input)).detach()
                alpha = E/torch.max(torch.abs(input)).detach()
            else:
                E= alpha*torch.max(torch.abs(input)).detach()
        else:
            alpha = E/torch.max(torch.abs(input)).detach()
        x = torch.clamp(input, min = -E, max = E)
        input_norm = x/E
        quantize = uniform_quantize(k-1, input_norm)
        #import pdb;pdb.set_trace()
        out = (quantize)
    return out, E, alpha

def quantize_fn_bit_slice(k, input, E, alpha, width):
    if k == 32:
        out = input
        E=1
        return out, E, None
    else:
        #import pdb;pdb.set_trace()
        if E is None:
            if alpha is None:
                #quantile=1-20/input.shape[0]
                #E = input.abs().quantile(quantile, interpolation ='linear')
                E = torch.max(torch.abs(input)).detach()
                alpha = E/torch.max(torch.abs(input)).detach()
            else:
                E= alpha*torch.max(torch.abs(input)).detach()
        else:
            alpha = E/torch.max(torch.abs(input)).detach()
        if k == width:
            E_out = E
            x = torch.clamp(input, min = -E, max = E)
            input_norm = x/E
            quantize = uniform_quantize(k-1, input_norm)
            #import pdb;pdb.set_trace()
        else:
            x = torch.clamp(input, min = -E, max = E)
            input_norm = x/E
            quantize, E_out = uniform_quantize_bit_slice(k, input_norm, width, E)
        #import pdb;pdb.set_trace()
        #offset_vat = 5/(2**k)
        out = (quantize)/(2**k)
        return out, E_out*(2**k), alpha
