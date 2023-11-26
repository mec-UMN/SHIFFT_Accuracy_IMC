from param import types,fft_size,xbar_size, uniform, weight_width, in_width, cell_prec,adc_bits, roll, quant, first_term, in_split,vat, rate
from fft import conv2d_Q_new_fn, conv2d_Q_bit_slice_fn,conv2d_Q_val_fn,weight_quantize
import numpy as np
from Utils import shift_add_input, shift_add, accuracy, accuracy_max
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_main():
    def __init__(self,weight_cos,weight_sin):
		#import pdb;pdb.set_trace()
        super(conv_main, self).__init__()
        self.weight_cos = weight_cos
        self.weight_sin = weight_sin

    def sync(self,E_adc_cos_new, E_adc_sin_new, E_adc_cos,  E_adc_sin, sign_prob, weight_cos_quant, weight_sin_quant, alpha_cos,alpha_sin, E_weight_cos, E_weight_sin, E_adc_cos_slice,E_adc_sin_slice, basicblock_no_vat_pre_cos, basicblock_no_vat_pre_sin):
        self.E_adc_cos_new=E_adc_cos_new
        self.E_adc_sin_new=E_adc_sin_new
        self.E_adc_cos=E_adc_cos
        self.E_adc_sin=E_adc_sin
        self.alpha_cos=alpha_cos
        self.alpha_sin=alpha_sin
        self.sign_prob= sign_prob
        self.weight_cos_quant =weight_cos_quant
        self.weight_sin_quant = weight_sin_quant
        self.E_weight_cos = E_weight_cos
        self.E_weight_sin = E_weight_sin
        self.E_adc_cos_slice=E_adc_cos_slice
        self.E_adc_sin_slice=E_adc_sin_slice
        self.basicblock_no_vat_pre_cos = basicblock_no_vat_pre_cos
        self.basicblock_no_vat_pre_sin = basicblock_no_vat_pre_sin
        

    def forward(self, input_fft_cos_up, input_fft_sin_up, j, mode, max_fft):
        if mode:
            E=max_fft*self.E_adc_cos_new[j]
            #import pdb;pdb.set_trace()
        else:
            E= None
     # Find E for each run or #find the precision corresponding to the updated E
        #import pdb;pdb.set_trace()
        if in_split ==0:
            self.basicblock_no_vat_pre_cos[j], self.E_adc_cos[j], self.alpha_cos[j] = conv2d_Q_bit_slice_fn(input_fft_cos_up[0], self.weight_cos_quant[j], E_weights=self.E_weight_cos, adc_bits=adc_bits[j], xbar_size=xbar_size[j],quant = quant, first_term=first_term[j], uniform=uniform[j], E=E, alpha = None, w_width=weight_width) 
            self.basicblock_no_vat_pre_sin[j] , self.E_adc_sin[j], self.alpha_sin[j] = conv2d_Q_bit_slice_fn(input_fft_sin_up[0], self.weight_sin_quant[j], E_weights=self.E_weight_sin, adc_bits=adc_bits[j], xbar_size=xbar_size[j], quant = quant, first_term=first_term[j], uniform=uniform[j], E=E, alpha = None, w_width=weight_width)
        else:
            for k in range(0, in_width):
            #find the appropritate E
            #import pdb;pdb.set_trace()
                basicblock_no_vat_slice_cos, self.E_adc_cos_slice[j], self.alpha_cos[j] = conv2d_Q_bit_slice_fn(input_fft_cos_up[in_width -1 - k],  self.weight_cos_quant[j], E_weights=self.E_weight_cos, adc_bits=adc_bits[j], xbar_size=xbar_size[j], quant = quant, first_term=first_term[j], uniform=uniform[j], E=E, alpha = None, w_width=weight_width) 
                basicblock_no_vat_slice_sin, self.E_adc_sin_slice[j], self.alpha_sin[j] = conv2d_Q_bit_slice_fn(input_fft_sin_up[in_width-1 - k],  self.weight_sin_quant[j], E_weights=self.E_weight_sin, adc_bits=adc_bits[j], xbar_size=xbar_size[j], quant = quant, first_term=first_term[j], uniform=uniform[j], E=E, alpha = None, w_width=weight_width)
                if   basicblock_no_vat_slice_cos.unique().size()[0]>2**(adc_bits[j]+fft_size/xbar_size[j]):
                    import pdb;pdb.set_trace()
                if k ==0:
                    self.basicblock_no_vat_pre_cos[j]=basicblock_no_vat_slice_cos
                    self.basicblock_no_vat_pre_sin[j]=basicblock_no_vat_slice_sin
                else:
                    #import pdb;pdb.set_trace()
                    self.basicblock_no_vat_pre_cos[j] = shift_add_input(self.basicblock_no_vat_pre_cos[j],basicblock_no_vat_slice_cos, k, in_width)
                    self.basicblock_no_vat_pre_sin[j] = shift_add_input(self.basicblock_no_vat_pre_sin[j],basicblock_no_vat_slice_sin, k, in_width)
                if k ==0:
                    self.E_adc_cos[j] = self.E_adc_cos_slice[j]*self.sign_prob
                    self.E_adc_sin[j] = self.E_adc_sin_slice[j]*self.sign_prob
                else:
                    self.E_adc_cos[j] = np.maximum (self.E_adc_cos_slice[j]*self.sign_prob,self.E_adc_cos[j])
                    self.E_adc_sin[j] = np.maximum (self.E_adc_sin_slice[j]*self.sign_prob,self.E_adc_sin[j])
        #print(self.E_adc_cos[0])
        #import pdb;pdb.set_trace()

        return self.basicblock_no_vat_pre_cos, self.basicblock_no_vat_pre_sin,self.E_adc_cos, self.E_adc_sin

class conv_grad(torch.autograd.Function):
    def __init__(self, mul, prec_mean_no_vat_cos, prec_mean_no_vat_sin, prec_stddev_no_vat_cos, prec_stddev_no_vat_sin,  prec_SQNR_no_vat_cos,  prec_SQNR_no_vat_sin):
		#import pdb;pdb.set_trace()
        super(conv_grad, self).__init__()
        self.mul = mul
        self.prec_mean_no_vat_cos=prec_mean_no_vat_cos
        self.prec_mean_no_vat_sin=prec_mean_no_vat_sin
        self.prec_stddev_no_vat_cos=prec_stddev_no_vat_cos
        self.prec_stddev_no_vat_sin=prec_stddev_no_vat_sin
        self.prec_SQNR_no_vat_cos=prec_SQNR_no_vat_cos
        self.prec_SQNR_no_vat_sin=prec_SQNR_no_vat_sin 

    def sync(self,basicblock_no_vat_pre_cos, basicblock_no_vat_pre_sin, indices_sin, indices_cos,basicblock_no_vat_cos, basicblock_no_vat_sin):
        self.basicblock_no_vat_pre_cos = basicblock_no_vat_pre_cos
        self.basicblock_no_vat_pre_sin = basicblock_no_vat_pre_sin
        self.indices_sin=indices_sin
        self.indices_cos=indices_cos
        self.basicblock_no_vat_cos=basicblock_no_vat_cos
        self.basicblock_no_vat_sin=basicblock_no_vat_sin

    def forward(self, i, j, E_input, basicblock_val_cos, basicblock_val_sin):
        if self.mul[j]>1:
        #print(basicblock_val_cos.data)
        #import pdb;pdb.set_trace()
        #test=torch.range(1,2*(fft_size)-1,2, dtype=int)
        #test1=torch.range(0,2*(fft_size)-2,2, dtype=int)
        #basicblock_no_vat_cos[:,j] = torch.add(basicblock_no_vat_pre_cos[j].index_select(0,test),basicblock_no_vat_pre_cos[j].index_select(0,test1))*E_input
        #basicblock_no_vat_sin[:,j] = torch.add(basicblock_no_vat_pre_sin[j].index_select(0,test),basicblock_no_vat_pre_sin[j].index_select(0,test1))*E_input
            #import pdb;pdb.set_trace()
            x=self.basicblock_no_vat_pre_cos[j].clone()*E_input
            y=self.basicblock_no_vat_pre_sin[j].clone()*E_input
            basicblock_no_vat_cos = shift_add(x, weight_width,cell_prec[j])
            basicblock_no_vat_sin = shift_add(y, weight_width,cell_prec[j])
        else:
            #import pdb;pdb.set_trace()
            basicblock_no_vat_cos= (self.basicblock_no_vat_pre_cos[j])*E_input 
            basicblock_no_vat_sin = (self.basicblock_no_vat_pre_sin[j])*E_input
        if roll[j] == True:
            basicblock_no_vat_sin = basicblock_no_vat_sin[self.indices_sin.argsort()]
            basicblock_no_vat_cos = basicblock_no_vat_cos[self.indices_cos.argsort()]
        """
        if vat[j]:
            if i == 0:
                fix_vat_cos[:,j] = basicblock_no_vat_cos[:,j] - basicblock_val_cos
                fix_vat_sin[:,j] = basicblock_no_vat_sin[:,j] - basicblock_val_sin
            else:
                #import pdb;pdb.set_trace()
                basicblock_no_vat_cos[:,j] = basicblock_no_vat_cos[:,j] - fix_vat_cos[:,j]
                basicblock_no_vat_sin[:,j] = basicblock_no_vat_sin[:,j] - fix_vat_sin[:,j]
        """
        #import pdb;pdb.set_trace()
        """
        fix_vat_cos.append((basicblock_no_vat_cos[:,j]/basicblock_val_cos).median())
        fix_vat_sin.append((basicblock_no_vat_sin[1:fft_size,j]/basicblock_val_sin[1:fft_size]).median())
        basicblock_no_vat_cos[:,j] = basicblock_no_vat_cos[:,j]/fix_vat_cos[j]
        basicblock_no_vat_sin[:,j] = basicblock_no_vat_sin[:,j]/fix_vat_sin[j]
        offset_cos.append((basicblock_no_vat_cos[:,j]-basicblock_val_cos).mean())
        offset_sin.append((basicblock_no_vat_sin[:,j]-basicblock_val_sin).mean())
        """
        self.prec_mean_no_vat_cos[i][j], self.prec_stddev_no_vat_cos[i][j], self.prec_SQNR_no_vat_cos[i][j]= accuracy_max(basicblock_no_vat_cos.data[1:fft_size], basicblock_val_cos.data[1:fft_size], adc_bits[j])
        self.prec_mean_no_vat_sin[i][j], self.prec_stddev_no_vat_sin[i][j], self.prec_SQNR_no_vat_sin[i][j]= accuracy_max(basicblock_no_vat_sin.data[1:fft_size], basicblock_val_sin.data[1:fft_size], adc_bits[j])
        """
        if i < runs:
            #import pdb;pdb.set_trace()
            x_df = pd.DataFrame(np.array([[E_adc_cos[j][0][0], alpha_cos[j].abs().mean().item(), torch.std_mean(alpha_cos[j].abs())[0].item()]]))
            x_df.to_csv('E_adc_cos_distr.csv', mode='a', index=False, header=False)
            x_df = pd.DataFrame(np.array([[E_adc_sin[j][0][0], alpha_sin[j].abs().mean().item(), torch.std_mean(alpha_sin[j].abs())[0].item()]]))
            x_df.to_csv('E_adc_sin_distr.csv', mode='a', index=False, header=False)
        """
        return  self.prec_mean_no_vat_cos, self.prec_mean_no_vat_cos,self.prec_SQNR_no_vat_cos, self.prec_SQNR_no_vat_sin

def E_update(E_adc_cos, E_adc_sin,E_adc_cos_new,E_adc_sin_new,i, mode, alpha_cos, alpha_sin):
    if mode:
        # Find minimum E across runs
        if i == 0: 
            E_adc_cos_new = E_adc_cos
            E_adc_sin_new = E_adc_sin
    else:
        E_adc_cos_new = E_adc_cos_new*(1-alpha_cos*rate)
        E_adc_sin_new = E_adc_sin_new*(1-alpha_sin*rate)
    return E_adc_cos_new, E_adc_sin_new