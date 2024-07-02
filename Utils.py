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
import csv
from quant import quantize_fn_uniform
from param import in_quant, runs_train, fft_size

def map_update(weight, map_type):
    if map_type == 0:
        weight_new = weight
        #input_new = input
        #import pdb;pdb.set_trace()
    else:
        fft_size = weight.shape[0]
        #import pdb;pdb.set_trace()
        weight_new = weight.roll(int(fft_size/(2*map_type)),0)
        #input_new = input.roll(int(fft_size/(2*map_type)))
        #import pdb;pdb.set_trace()
    return weight_new


def accuracy_classify(output, target, dev):
    """Computes the precision@k for the specified values of k"""
    N = target.size(0)
    res=0
    for i in range(N):
        #import pdb;pdb.set_trace()
        if (abs(output[i]) > abs(target[i]*(1-0.5*dev))) and (abs(output[i]) < abs(target[i]*(1+0.5*dev))) :
            res += 1
    res = res*100/N
    return res

def accuracy(output, target, n):
    """Computes the precision@k for the specified values of k"""
    #import pdb;pdb.set_trace()
    error = torch.div((output - target)*100, target).abs()
    mean = error.mean()
    stddev = error.std()
    SQNR = 10*torch.log10(torch.div(target.square().mean(),(output - target).square().mean()))
    #import pdb;pdb.set_trace()
    return mean.item(), stddev.item(), SQNR.item()

def accuracy_max(output, target, n):
    """Computes the precision@k for the specified values of k"""
    #import pdb;pdb.set_trace()
    MAE=(output - target).abs().sum()/(output.abs().max()*fft_size)
    error = ((output - target)*100*(2**(n-1))/target.abs().max()).abs()
    #error = torch.div((output - target)*100, target).abs()
    mean = error.mean()
    stddev = error.std()
    SQNR = 10*torch.log10(torch.div(target.square().mean(),(output - target).square().mean()))
    #import pdb;pdb.set_trace()
    return mean.item(), stddev.item(), SQNR.item(), MAE.item()


def shift_add(input, wg_width, w_bit):
    indices = math.ceil(wg_width/w_bit)
    size = int(input.shape[0]/indices)
    index = torch.range(0,size-1,dtype=int)
    #import pdb;pdb.set_trace()
    #out = torch.add(input[index]*(2**(indices-1)*w_bit),input[index+1]) #only applicable for indices=2
    #import pdb;pdb.set_trace()
    output = torch.add(input[index]*(2**((indices-1)*w_bit)),input[index+size])
    #output = torch.zeros([size], dtype=torch.float64)
    #for i in range(input.shape[0]):
        #import pdb;pdb.set_trace()
        #output[i//indices] = output[i//indices] + input[i]*(2**((indices-1-(i%indices))*w_bit))
    #print(input[0:10])
    #import pdb;pdb.set_trace()
    return output

def shift_add_11(input, wg_width, w_bit):
    indices = math.ceil(wg_width/w_bit)
    mul =[11,1]
    size = int(input.shape[0]/indices)
    #import pdb;pdb.set_trace()
    output = torch.zeros([size], dtype=torch.float64)
    for i in range(input.shape[0]):
        #import pdb;pdb.set_trace()
        output[i//indices] = output[i//indices] + input[i]*(mul[i%indices])
    #print(input[0:10])
    #import pdb;pdb.set_trace()
    return output

def shift_add_input(out, input, index, in_width):
    #import pdb;pdb.set_trace()
    out = torch.add(out,input*(2**(index)))
    #import pdb;pdb.set_trace()
    return out

def input_generate(size, k, src, in_width, split,adc_quant_factor, E):
    input = torch.ones([size], dtype=torch.float64, requires_grad=True)
    #k = None
    if k is None:
        #for i in range (0, size):
           #input[i] = random.randrange(1, 2**in_width)
           #input[i] = 2**(in_width-1)-1
        E_input =1
        input_unquantized =input*1/128
        input_slice,scale_input, scale_E =dec2bin(input_unquantized, in_width, split)
    elif k ==1:
        E_input =1
        input_unquantized = torch.randn(size).to(torch.float64)/3
        #import pdb;pdb.set_trace()
        input_slice,scale_input, scale_E =dec2bin(input_unquantized, in_width, split)
    else:
        sample_rate = src.size()[0]
        n = math.floor(sample_rate/size)
        arr = n*torch.range(0,size-1).to(torch.int)
        input_unquantized = src[arr]  
       #input = input_unquantized
       #E_input =1
        if in_quant:
           input, E_input, alpha_input = quantize_fn_uniform(in_width, input_unquantized, E, None)
        else:
           input=input_unquantized
           E_input =1
        Max_fft = input.abs().sum().item()*adc_quant_factor
        input_slice, scale_input, scale_E = dec2bin(input, in_width, split)
        #import pdb;pdb.set_trace()
    """
    x_df = pd.DataFrame(input)
    x_df.to_csv('./Data/input_quant.csv')
    x_df = pd.DataFrame(input_unquantized)
    x_df.to_csv('./Data/input.csv')
    """
    return input, E_input*scale_E, input_unquantized, input_slice, Max_fft

def weight_generate_imc(size, wg_width, w_bit):
    weight_ac = torch.zeros([size, size], dtype=torch.float64)
    indices = math.ceil(wg_width/w_bit)
    weight = torch.zeros([size, size*indices], dtype=torch.float64)
    for i in range (0, size):
        for j in range(0,size):
            weight_ac[i,j] = np.cos(2*math.pi*i*j/size)
            for k in range(indices):
                weight[i,indices*j+k] = math.fmod((weight_ac[i,j]*(2**(wg_width)-1)/(2**(k*w_bit))),(2**w_bit))
    weight=weight/(2**(wg_width)-1)
    #import pdb;pdb.set_trace()
    return weight

def weight_generate(size, cos):
    weight = torch.zeros([size, size], dtype=torch.float64)
    for i in range (0, size):
        for j in range(0,size):
            if cos == True:
                weight[i,j] = math.cos(2*math.pi*i*j/size)
            else:
                weight[i,j] = math.sin(2*math.pi*i*j/size)
    return weight

def offset(weight, in_width, adc_bits,in_split):
    size=weight.shape[0]
    out=[]
    no_iter = np.array(adc_bits).size
    if in_split==0:
        for i in range(0, no_iter):
            out.append(torch.zeros([size], dtype=torch.float64))
    else:
        input = torch.ones([size], dtype=torch.float64)
        #import pdb;pdb.set_trace()
        for i in range(0, no_iter):
            #import pdb;pdb.set_trace()
            offset=torch.matmul(input,weight)*(2**(in_width-1))
            quant_off=quantize_fn_uniform(adc_bits[i],offset,None, None)
            out.append(quant_off[0]*quant_off[1])
            #out.append(offset)
    return out

def prob_sign(input):
    out=input.clone()
    size=input.size(dim=0)
    out[input>=0] =0
    neg = out.count_nonzero().item()
    prob = neg/size
    #print("neg:",neg)
    return prob 

def fft_symm(input, cos):
    out=input
    size=input.size()[0]
    for i in range(1,size):
        if cos == True:
            out[i]+= input[size-i]
        else:
            out[i]-= input[size-i]
        out[i]=out[i]/2
    #import pdb;pdb.set_trace()
    return out
  
def dec2bin(x,n, split):
    if split ==0:
        scale_list=[]
        delta=1
        out = []
        for i in range(n-1):
            out.append(x.clone())
    else:
        y = x.clone()
        sign =x.clone()
        out = []
        scale_list = []
        delta = 1.0/(2**(n-1))
        x_int = (x+1)/delta
        #import pdb;pdb.set_trace()
        base = 2**(n)

        y[x_int>=0] = 0
        y[x_int< 0] = 1
        rest = x_int 
        #out.append(y.clone())
        scale_list.append(-base*delta)
        for i in range(n):
            base = base/2
            y[rest>=base] = 1
            y[rest<base]  = 0
            rest = rest - base * y
            out.append(y.clone())
            #import pdb;pdb.set_trace()
            scale_list.append(base * delta)
        #out_arr=np.array(out)
        #out_re=out_arr.reshape(1, out_arr.size)
        #import pdb;pdb.set_trace()
        #print(out[0].unique())
    return out,scale_list, delta

def plot_curve(self,j):
    title = 'SQNR vs E'
    dpi = 80  
    width, height = 1200, 800
    legend_fontsize = 10
    scale_distance = 48.8
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    x_axis = 0.1*np.array([i for i in range(runs_train)]) # runs
    y_axis = np.zeros(runs_train)

    plt.xlim(max(self.x),min(self.x))
    plt.ylim(-5, 25)
    interval_y = 1
    interval_x = -0.05
    plt.xticks(np.arange(max(self.x),min(self.x) + interval_x, interval_x))
    plt.yticks(np.arange(-5, 25 + interval_y, interval_y))
    plt.grid()
    plt.title(title, fontsize=20)
    plt.xlabel('E', fontsize=16)
    plt.ylabel('SQNR', fontsize=16)
    
    #import pdb;pdb.set_trace()		
    x_axis[:] = self.x
    y_axis[:] = self.prec_SQNR_no_vat_cos[:,j]
    plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-SQNR', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    """
    y_axis[:] = self.epoch_accuracy[:, 1]
    plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    
    y_axis[:] = self.epoch_losses[:, 0]
    plt.plot(x_axis, y_axis*50, color='g', linestyle=':', label='train-loss-x50', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    y_axis[:] = self.epoch_losses[:, 1]
    plt.plot(x_axis, y_axis*50, color='y', linestyle=':', label='valid-loss-x50', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)
    """
    save_path="./Figures/SQNR.png"
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print ('---- save figure {} into {}'.format(title, save_path))
    plt.close(fig)

