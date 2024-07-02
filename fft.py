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
from quant import quantize_fn, quantize_fn_bit_slice, quantize_fn_uniform
from vat import add_vat,add_vat_new, add_vat_denoise
from param import wg_quant, adc_quant

def conv2d_Q_new_fn(input, weight, w_bit, adc_bits, xbar_size, vat, quant, first_term, uniform, E, alpha):
    x=input
    if quant:
        weight_q_no_vat, E_weights, alpha_weights= quantize_fn(w_bit, weight, None, None)
    else:
        weight_q_no_vat = weight
    if vat:
        weight_q=add_vat(w_bit,weight_q_no_vat)
        #import pdb;pdb.set_trace()
    else:
        weight_q = weight_q_no_vat
    fft_size = x.shape[0]
    #import pdb;pdb.set_trace()
    outputRegular_RRAM = torch.matmul(x,weight_q)
    #print(x)
    #print(weight)

    res = torch.zeros_like(outputRegular_RRAM)
    #print("---------mapping to RRAM crossbar---------")
    weights_shape = list(weight_q.size())
    N_rows_Xbar = int(weights_shape[0]/xbar_size)
    N_cols_Xbar = int(weights_shape[1]/xbar_size)
    E_adc_arr = np.zeros((N_rows_Xbar, N_cols_Xbar))
    alpha_arr = np.zeros((N_rows_Xbar, N_cols_Xbar))
    start_col = 0 
    end_col = xbar_size
    j=0
    #import pdb;pdb.set_trace()
    while (end_col<=weights_shape[1]):
        start_row = 0
        end_row = xbar_size
        res_temp_2=0
        i=0
        E_adc=0
        while (end_row<=weights_shape[0]):
            x_temp = x[start_row:end_row]
            weight_temp = weight_q[start_row:end_row, start_col:end_col]
            res_temp = torch.matmul(x_temp,weight_temp)
            #import pdb;pdb.set_trace()
            if quant:
                if alpha is None:
                    alpha_adc = None
                else:
                    alpha_adc=alpha[i,j]
                if E is None:
                   E_adc = None
                else:
                    E_adc=np.mean(E[:,j])
                #import pdb;pdb.set_trace()
                if not first_term:
                    if start_col == 0:
                        if uniform:
                            res_temp, E_adc, alpha_adc= quantize_fn_uniform(adc_bits , res_temp[1:xbar_size], E_adc, None)
                        else:
                            res_temp, E_adc, alpha_adc= quantize_fn(adc_bits , res_temp[1:xbar_size], E_adc, None)
                        res_temp = torch.cat((torch.tensor([0]),res_temp), 0) 
                    else:
                        if uniform:
                            res_temp, E_adc, alpha_adc= quantize_fn_uniform(adc_bits , res_temp, E_adc, None)
                        else:
                            res_temp, E_adc, alpha_adc= quantize_fn(adc_bits , res_temp, E_adc, None)
                else:
                    if uniform:
                        res_temp, E_adc, alpha_adc= quantize_fn_uniform(adc_bits , res_temp, E_adc, None)
                    else:
                        res_temp, E_adc, alpha_adc= quantize_fn(adc_bits , res_temp, E_adc, None)
            #print(res_temp*E_adc)
            #print(res_temp_2*E_adc)
            #print(x_temp)
            #print(weight_temp*E_weights)
            E_adc_arr[i][j] = E_adc
            alpha_arr[i][j]=alpha_adc
            if quant:
                res_temp_3 = torch.add(res_temp_2, res_temp*E_adc)
            else:
                res_temp_3 = torch.add(res_temp_2, res_temp)
            res_temp_2 = torch.clone(res_temp_3)
            #import pdb;pdb.set_trace()
            #print(res_temp_2[0])
            start_row += xbar_size
            end_row += xbar_size
            i+=1
        res[start_col:end_col] = res_temp_2 
        start_col += xbar_size
        end_col += xbar_size
        j+=1
    #import pdb;pdb.set_trace()
    if quant:
        output = res
    else:
        output = res	
    return output, E_adc_arr, alpha_arr

def conv2d_Q_val_fn(input, weight):
    x=input
    weight_q=weight
    #import pdb;pdb.set_trace()
    outputRegular_RRAM = torch.matmul(x,weight_q)
    output = outputRegular_RRAM        
    return output

def weight_quantize(weight_orgin, weight_roll, w_bit, w_width, vat, quant, roll, types):
    weight_q = []
    for i in range(0, types):
        if roll[i] == False:
            weight=weight_orgin
        else:
            weight=weight_roll
        if wg_quant:
            weight_q_no_vat, E_weights, alpha_weights= quantize_fn_bit_slice(w_bit[i], weight, None, None, w_width)
        else:
            weight_q_no_vat = weight
            E_weights=1
        #import pdb;pdb.set_trace()
        if vat[i]:
            weight_q.append(add_vat_denoise(w_bit[i],weight_q_no_vat))
            #import pdb;pdb.set_trace()
        else:
            weight_q.append(weight_q_no_vat)
    return weight_q, E_weights

def conv2d_Q_bit_slice_fn(input, weight_q, E_weights, adc_bits, xbar_size, quant, first_term, uniform, E, alpha, w_width):
    x=input
    quant_fn=quantize_fn(adc_bits)
    #fft_size = x.shape[0]
    #import pdb;pdb.set_trace()
    outputRegular_RRAM = torch.matmul(x,weight_q)
    #print(x)
    #print(weight)
    #import pdb;pdb.set_trace()

    res = torch.zeros_like(outputRegular_RRAM)
    #print("---------mapping to RRAM crossbar---------")
    input_shape = list(input.size())
    weights_shape = list(weight_q.size())
    N_rows_Xbar = int(weights_shape[0]/xbar_size)
    N_cols_Xbar = int(weights_shape[1]/xbar_size)
    E_adc_arr = np.zeros((N_rows_Xbar, N_cols_Xbar))
    alpha_arr = np.zeros((N_rows_Xbar, N_cols_Xbar))
    start_col = 0 
    end_col = xbar_size
    j=0
    #import pdb;pdb.set_trace()
    while (end_col<=weights_shape[1]):
        #import pdb;pdb.set_trace()
        start_row = 0
        end_row = xbar_size
        res_temp_2=0
        i=0
        E_adc=0
        while (end_row<=weights_shape[0]):
            x_temp = x[start_row:end_row]
            weight_temp = weight_q[start_row:end_row, start_col:end_col]
            res_temp = torch.matmul(x_temp,weight_temp)
            #import pdb;pdb.set_trace()
            if adc_quant:
                if (start_col > input_shape[0]):
                    signed = True
                else:
                    signed =True
                if alpha is None:
                    alpha_adc = None
                else:
                    alpha_adc=alpha[i,j]
                #import pdb;pdb.set_trace()
                if E is None:
                   E_adc = None
                else:
                    E_adc=np.mean(E[:,j])
                #import pdb;pdb.set_trace()
                if not first_term:
                    if start_col == 0:
                        if uniform:
                            res_temp, E_adc, alpha_adc= quantize_fn_uniform(adc_bits , res_temp[1:xbar_size], E_adc, None)
                        else:
                            res_temp, E_adc, alpha_adc= quant_fn.forward(res_temp[1:xbar_size], E_adc, None, signed)
                        res_temp = torch.cat((torch.tensor([0]),res_temp), 0) 
                    else:
                        if uniform:
                            res_temp, E_adc, alpha_adc= quantize_fn_uniform(adc_bits , res_temp, E_adc, None)
                        else:
                            res_temp, E_adc, alpha_adc= quant_fn.forward(res_temp, E_adc, None, signed)
                else:
                    if uniform:
                        res_temp, E_adc, alpha_adc= quantize_fn_uniform(adc_bits , res_temp, E_adc, None, True)
                        #print(res_temp)
                        #print(E_adc)
                    else:
                        res_temp, E_adc, alpha_adc= quant_fn.forward( res_temp, E_adc, None, signed)
            #print(res_temp*E_adc)
            #print(res_temp_2*E_adc)
            #print(x_temp)
            #print(weight_temp*E_weights)
            #print(adc_bits)
            #if res_temp.unique().size()[0]> 2**adc_bits:
                #import pdb;pdb.set_trace()
                E_adc_arr[i][j] = E_adc
                alpha_arr[i][j]=alpha_adc
                res_temp_3 = torch.add(res_temp_2, res_temp*E_adc)
            else:
                res_temp_3 = torch.add(res_temp_2, res_temp)
            res_temp_2 = torch.clone(res_temp_3)
            #import pdb;pdb.set_trace()
            #print(res_temp_2[0])
            start_row += xbar_size
            end_row += xbar_size
            i+=1
        res[start_col:end_col] = res_temp_2 
        start_col += xbar_size
        end_col += xbar_size
        j+=1
    #import pdb;pdb.set_trace()
    #print(E_adc_arr)
    if wg_quant:
        output = res*E_weights
    else:
        output = res
    return output, E_adc_arr, outputRegular_RRAM*E_weights