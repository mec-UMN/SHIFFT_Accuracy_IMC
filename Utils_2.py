import numpy as np
import math
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import os # accessing directory structure


from fft import conv2d_Q_new_fn, conv2d_Q_bit_slice_fn,conv2d_Q_val_fn,weight_quantize
from Utils import weight_generate, accuracy, weight_generate_imc, shift_add, shift_add_11, map_update, shift_add_input, input_generate, offset, prob_sign,fft_symm
from param import types,fft_size,xbar_size, uniform, weight_width, in_width, cell_prec,adc_bits, roll, quant, first_term, in_split,vat

class val_generate():
    def __init__(self,weight_cos,weight_sin):
        #import pdb;pdb.set_trace()
        super(val_generate, self).__init__()
        self.weight_cos = weight_cos
        self.weight_sin = weight_sin

    def forward(self, input_fft_unquant):
        basicblock_val_cos = conv2d_Q_val_fn(input_fft_unquant, self.weight_cos)
        basicblock_val_sin = conv2d_Q_val_fn(input_fft_unquant, self.weight_sin)
        """
        x_df = pd.DataFrame(basicblock_val_cos.data)
        x_df.to_csv('./Data/cosine.csv')	
        x_df = pd.DataFrame(basicblock_val_sin.data)
        x_df.to_csv('./Data/sine.csv')
        """
        return basicblock_val_cos, basicblock_val_sin


def weight_generate_main(fft_size):
    weight_cos = weight_generate(fft_size, cos=True)
    weight_sin = weight_generate(fft_size, cos=False)
    """
    x_df = pd.DataFrame(weight_cos[0:512][0:512])
    x_df.to_csv('./Data/Weight_cos.csv')
    x_df = pd.DataFrame(weight_sin[0:512][0:512])
    x_df.to_csv('./Data/Weight_sin.csv')
    """
    return weight_cos, weight_sin

def initialize_params(self, runs):
    self.E_adc_cos=[]
    self.E_adc_sin=[]
    self.E_adc_cos_slice=[]
    self.E_adc_sin_slice=[]
    self.E_adc_cos_new=[]
    self.E_adc_sin_new=[]
    self.alpha_cos =[]
    self.alpha_sin=[]
    self.mul = np.zeros(types)
    self.basicblock_no_vat_pre_cos=[]
    self.basicblock_no_vat_pre_sin=[]
    self.basicblock_no_vat_slice_cos=[]
    self.basicblock_no_vat_slice_sin=[]
    
    self.prec_mean_no_vat_cos = np.zeros((runs,types))
    self.prec_stddev_no_vat_cos = np.zeros((runs,types))
    self.prec_SQNR_no_vat_cos = np.zeros((runs,types))
    self.prec_mean_no_vat_sin = np.zeros((runs,types))
    self.prec_stddev_no_vat_sin = np.zeros((runs,types))
    self.prec_SQNR_no_vat_sin =  np.zeros((runs,types))
    
    self.indices_sin = np.zeros((fft_size, 1))
    self.indices_cos = np.zeros((fft_size, 1))

    for k in range(0,types):
        num_row_col = int(fft_size/xbar_size[k])
        self.mul[k] = math.ceil(weight_width/cell_prec[k]) 
        self.E_adc_cos.append(np.zeros((num_row_col, int(num_row_col* self.mul[k]))))
        self.E_adc_sin.append(np.zeros((num_row_col, int(num_row_col* self.mul[k]))))
        self.E_adc_cos_slice.append(np.zeros((num_row_col, int(num_row_col* self.mul[k]))))
        self.E_adc_sin_slice.append(np.zeros((num_row_col, int(num_row_col* self.mul[k]))))
        self.E_adc_cos_new.append(np.ones((num_row_col, int(num_row_col* self.mul[k]))))
        self.E_adc_sin_new.append(np.ones((num_row_col, int(num_row_col* self.mul[k]))))
        self. alpha_cos.append(np.zeros((num_row_col, num_row_col)))
        self.alpha_sin.append(np.zeros((num_row_col, num_row_col)))
        #map[1,k] = num_row_col
        self.basicblock_no_vat_pre_cos.append(torch.zeros([int(fft_size* self.mul[k])], dtype=torch.float64, requires_grad=True))
        self.basicblock_no_vat_pre_sin.append(torch.zeros([int(fft_size* self.mul[k])], dtype=torch.float64,requires_grad=True))
        self.basicblock_no_vat_slice_cos.append(torch.zeros([int(fft_size* self.mul[k])], dtype=torch.float64, requires_grad=True))
        self.basicblock_no_vat_slice_sin.append(torch.zeros([int(fft_size* self.mul[k])], dtype=torch.float64,requires_grad=True))
    self.basicblock_no_vat_cos = torch.zeros([fft_size,types], dtype=torch.float64,requires_grad=True)
    self.basicblock_no_vat_sin = torch.zeros([fft_size,types], dtype=torch.float64,requires_grad=True)
    self.fix_vat_cos=[]
    self.fix_vat_sin=[]
    self.offset_cos =[]
    self.offset_sin=[]
    #weight_imc = weight_generate_imc(fft_size, weight_width, cell_prec)
    """
    x_df = pd.DataFrame(self.E_adc_cos[0])
    x_df.to_csv('./Data/E_adc_cos.csv')
    x_df = pd.DataFrame(self.E_adc_sin[0])
    x_df.to_csv('./Data/E_adc_sin.csv')
    x_df = pd.DataFrame(self.E_adc_cos[0])
    x_df.to_csv('./Data/E_adc_cos_distr.csv')
    x_df = pd.DataFrame(self.E_adc_sin[0])
    x_df.to_csv('./Data/E_adc_sin_distr.csv')
    """
    #import pdb;pdb.set_trace()
    return self

def initialize_params_test(self, runs, E_cos, E_sin):
    self.E_adc_cos=[]
    self.E_adc_sin=[]
    self.E_adc_cos_slice=[]
    self.E_adc_sin_slice=[]
    self.E_adc_cos_new=[]
    self.E_adc_sin_new=[]
    self.alpha_cos =[]
    self.alpha_sin=[]
    self.mul = np.zeros(types)
    self.basicblock_no_vat_pre_cos=[]
    self.basicblock_no_vat_pre_sin=[]
    self.basicblock_no_vat_slice_cos=[]
    self.basicblock_no_vat_slice_sin=[]
    
    self.prec_mean_no_vat_cos = np.zeros((runs,types))
    self.prec_stddev_no_vat_cos = np.zeros((runs,types))
    self.prec_SQNR_no_vat_cos = np.zeros((runs,types))
    self.prec_mean_no_vat_sin = np.zeros((runs,types))
    self.prec_stddev_no_vat_sin = np.zeros((runs,types))
    self.prec_SQNR_no_vat_sin =  np.zeros((runs,types))
    
    self.indices_sin = np.zeros((fft_size, 1))
    self.indices_cos = np.zeros((fft_size, 1))

    for k in range(0,types):
        num_row_col = int(fft_size/xbar_size[k])
        self.mul[k] = math.ceil(weight_width/cell_prec[k]) 
        self.E_adc_cos.append(np.ones((num_row_col, int(num_row_col* self.mul[k]))))
        self.E_adc_sin.append(np.ones((num_row_col, int(num_row_col* self.mul[k]))))
        self.E_adc_cos_slice.append(np.zeros((num_row_col, int(num_row_col* self.mul[k]))))
        self.E_adc_sin_slice.append(np.zeros((num_row_col, int(num_row_col* self.mul[k]))))
        #self.E_adc_cos_new.append(np.ones((num_row_col, int(num_row_col* self.mul[k]))))
        #self.E_adc_sin_new.append(np.ones((num_row_col, int(num_row_col* self.mul[k]))))
        self. alpha_cos.append(np.zeros((num_row_col, num_row_col)))
        self.alpha_sin.append(np.zeros((num_row_col, num_row_col)))
        #map[1,k] = num_row_col
        self.basicblock_no_vat_pre_cos.append(torch.zeros([int(fft_size* self.mul[k])], dtype=torch.float64, requires_grad=True))
        self.basicblock_no_vat_pre_sin.append(torch.zeros([int(fft_size* self.mul[k])], dtype=torch.float64,requires_grad=True))
        self.basicblock_no_vat_slice_cos.append(torch.zeros([int(fft_size* self.mul[k])], dtype=torch.float64, requires_grad=True))
        self.basicblock_no_vat_slice_sin.append(torch.zeros([int(fft_size* self.mul[k])], dtype=torch.float64,requires_grad=True))
    self.basicblock_no_vat_cos = torch.zeros([fft_size,types], dtype=torch.float64,requires_grad=True)
    self.basicblock_no_vat_sin = torch.zeros([fft_size,types], dtype=torch.float64,requires_grad=True)
    self.fix_vat_cos=[]
    self.fix_vat_sin=[]
    self.offset_cos =[]
    self.offset_sin=[]
    self.E_adc_cos_new=E_cos
    self.E_adc_sin_new=E_sin
    #weight_imc = weight_generate_imc(fft_size, weight_width, cell_prec)
    """
    x_df = pd.DataFrame(self.E_adc_cos[0])
    x_df.to_csv('./Data/E_adc_cos.csv')
    x_df = pd.DataFrame(self.E_adc_sin[0])
    x_df.to_csv('./Data/E_adc_sin.csv')
    x_df = pd.DataFrame(self.E_adc_cos[0])
    x_df.to_csv('./Data/E_adc_cos_distr.csv')
    x_df = pd.DataFrame(self.E_adc_sin[0])
    x_df.to_csv('./Data/E_adc_sin_distr.csv')
    """
    #import pdb;pdb.set_trace()
    return self

def initialize_params_no_train(self):
    self.E_adc_cos_new = [None] * types
    self.E_adc_sin_new = [None] * types
    self.indices_cos = [None] * types
    self.indices_sin = [None] * types
    return self

def weight_quantize_main(self, basicblock_val_cos,basicblock_val_sin):
    #indices_sin = np.arange(0,fft_size,1)
    #indices_cos = np.arange(0,fft_size,1)
    sort_basicblock, self.indices_sin = basicblock_val_sin.abs().sort()
    sort_basicblock, self.indices_cos = basicblock_val_cos.abs().sort()
    weight_sin_map = self.weight_sin[:,self.indices_sin]
    weight_cos_map = self.weight_cos[:,self.indices_cos]
    self.weight_cos_quant, self.E_weight_cos= weight_quantize(self.weight_cos, weight_cos_map, cell_prec, weight_width, vat, quant, roll, types)
    self.weight_sin_quant, self.E_weight_sin= weight_quantize(self.weight_sin, weight_sin_map, cell_prec, weight_width, vat, quant, roll, types)
    #print(indices_cos)
    return self

def load_dataset_ecg(split, shuffle_files, data_dir, runs):
    nRowsRead = runs # specify 'None' if want to read whole file
    # mitbih_test.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
    if split == 'train':
        df1 = pd.read_csv('./dataset/ECG/mitbih_train.csv', delimiter=',', nrows = nRowsRead)
        df1.dataframeName = 'mitbih_train.csv'
    nRow, nCol = df1.shape
    print(f'There are {nRow} rows and {nCol} columns')
    df1=df1.head()
    ds=tf.convert_to_tensor(df1)
    #import pdb;pdb.set_trace()
    return ds