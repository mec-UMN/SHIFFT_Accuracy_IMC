import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import argparse
import sys
import shutil
import time
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core.utils import gcs_utils
from fft import conv2d_Q_new_fn, conv2d_Q_bit_slice_fn,conv2d_Q_val_fn,weight_quantize
from Utils import weight_generate, accuracy, weight_generate_imc, shift_add, shift_add_11, map_update, shift_add_input, input_generate, offset, prob_sign,fft_symm
from param import types,fft_size,xbar_size, uniform, weight_width, in_width, cell_prec,adc_bits, roll, quant, first_term, in_split,vat, runs_test, runs_train, train_mode, dataset, in_quant, wg_quant
from Utils_2 import val_generate, weight_generate_main,initialize_params,weight_quantize_main, initialize_params_no_train,load_dataset_ecg, initialize_params_test
from fft_Util import conv_main, E_update, conv_grad
from plot_Utils import plot_curve, plot_curve_prec


def model_train(dataset, args, runs_train):
	gcs_utils._is_gcs_disabled = True
	if dataset == "spoken_digit":
		ds = tfds.load('spoken_digit', split='train', shuffle_files=True, data_dir=args.data_path, try_gcs=False, as_supervised=True)
	elif dataset == "crema_d":
		ds = tfds.load('crema_d', split='train', shuffle_files=True, data_dir=args.data_path, try_gcs=False, as_supervised=True)
	elif dataset == "speech_commands":
		ds = tfds.load('speech_commands', split='train', shuffle_files=True, data_dir=args.data_path, try_gcs=False, as_supervised=True)
	elif dataset == "groove":
		ds = tfds.load('groove/full-16000hz', split='train', shuffle_files=True, data_dir=args.data_path, try_gcs=False, as_supervised=False)
	elif dataset == "nsynth":
		ds = tfds.load('nsynth', split='train', shuffle_files=True, data_dir=args.data_path, try_gcs=False, as_supervised=False)
	elif dataset =="ecg":
		ds=load_dataset_ecg(split='train', shuffle_files=True, data_dir=args.data_path, runs=runs_train)
	else:
		assert False, "Unknow dataset : {}".format(args.dataset)
	print(ds)
	if dataset !="ecg":
		assert isinstance(ds, tf.data.Dataset)
		ds = ds.take(runs_train)  # Only take a single example
	#import pdb;pdb.set_trace()
	return ds

def model_test(dataset,args):
	gcs_utils._is_gcs_disabled = True
	#import pdb;pdb.set_trace()
	if dataset == "crema_d":
		ds = tfds.load('crema_d', split='test', shuffle_files=True, data_dir=args.data_path, try_gcs=False, as_supervised=True)
	elif dataset == "speech_commands":
		ds = tfds.load('speech_commands', split='test', shuffle_files=True, data_dir=args.data_path, try_gcs=False, as_supervised=True)
	elif dataset == "groove":
		ds = tfds.load('groove/full-16000hz', split='test', shuffle_files=True, data_dir=args.data_path, try_gcs=False, as_supervised=False)
	elif dataset == "nsynth":
		ds = tfds.load('nsynth', split='test', shuffle_files=True, data_dir=args.data_path, try_gcs=False, as_supervised=False)
	else:
		assert False, "Unknow dataset : {}".format(args.dataset)
	assert isinstance(ds, tf.data.Dataset)
	print(ds)
	
	ds = ds.take(runs_test)  # Only take a single example
	#import pdb;pdb.set_trace()
	return ds

class train():
	def __init__(self):
		#import pdb;pdb.set_trace()
		super(train, self).__init__()
		weight_cos, weight_sin = weight_generate_main(fft_size)
		self.weight_cos = weight_cos
		self.weight_sin = weight_sin
		self.val_generate = val_generate(weight_cos, weight_sin)
		self=initialize_params(self, runs_train)
		self.conv_main = conv_main(self.weight_cos,self.weight_sin)
		self.conv_grad=conv_grad(self.mul, self.prec_mean_no_vat_cos, self.prec_mean_no_vat_sin, self.prec_stddev_no_vat_cos, self.prec_stddev_no_vat_sin,  self.prec_SQNR_no_vat_cos,  self.prec_SQNR_no_vat_sin)
		#import pdb;pdb.set_trace()
		self.dataset=dataset
		self.x=[]
	
	def forward(self, ds):
		delta_cos=0
		i=0
		#offset_cos = offset(weight_cos, in_width, adc_bits, in_split)
		#offset_sin = offset(weight_sin, in_width, adc_bits, in_split)
		for audio,label in tfds.as_numpy(ds):
			#import pdb;pdb.set_trace()
			#audio = np.array(list(audio_dict.items()))[0,1]
			print(type(audio),label,i)
			audio_torch = torch.from_numpy(audio)
			in_type = 2
			input_src = audio_torch.to(torch.float64)
			#import pdb;pdb.set_trace()
			input_fft, E_input, input_fft_unquant, input_fft_slice= input_generate(fft_size, in_type, input_src, in_width, in_split)
			input_fft_cos_up = input_fft_slice
			input_fft_sin_up = input_fft_slice
			basicblock_val_cos,basicblock_val_sin=self.val_generate.forward(input_fft_unquant)	
			if i ==0:
				self = weight_quantize_main(self, basicblock_val_cos, basicblock_val_sin) 
			#import pdb;pdb.set_trace()
			self.sign_prob =prob_sign(input_fft)
			for j in range(0,types):
				if i ==0:
					self.conv_main.sync(self.E_adc_cos_new, self.E_adc_sin_new, self.E_adc_cos,  self.E_adc_sin, self.sign_prob,self.weight_cos_quant, self.weight_sin_quant, self.alpha_cos, self.alpha_sin, self.E_weight_cos, self.E_weight_sin, self.E_adc_cos_slice, self.E_adc_sin_slice, self.basicblock_no_vat_pre_cos, self.basicblock_no_vat_pre_sin)
					self.conv_main.forward(input_fft_cos_up, input_fft_sin_up, j, False)
					self.E_adc_cos_new[j], self.E_adc_sin_new[j] = E_update( self.E_adc_cos[j],  self.E_adc_sin[j], self.E_adc_cos_new[j], self.E_adc_sin_new[j],i, True, self.prec_mean_no_vat_cos[i-1][j], self.prec_mean_no_vat_sin[i-1][j])
				#import pdb;pdb.set_trace()
				#print(basicblock_no_vat_slice_cos[0].unique())
				else:
					self.conv_main.sync(self.E_adc_cos_new, self.E_adc_sin_new, self.E_adc_cos,  self.E_adc_sin, self.sign_prob,self.weight_cos_quant, self.weight_sin_quant, self.alpha_cos, self.alpha_sin, self.E_weight_cos, self.E_weight_sin, self.E_adc_cos_slice, self.E_adc_sin_slice, self.basicblock_no_vat_pre_cos, self.basicblock_no_vat_pre_sin)
					self.conv_main.forward(input_fft_cos_up, input_fft_sin_up, j, True)
				self.conv_grad.sync(self.basicblock_no_vat_pre_cos, self.basicblock_no_vat_pre_sin, self.indices_sin, self.indices_cos, self.basicblock_no_vat_cos, self.basicblock_no_vat_sin)
				self.conv_grad.forward(i,j,E_input, basicblock_val_cos, basicblock_val_sin)
				#loss_cos = (basicblock_no_vat_cos).pow(2).sum()
				#import pdb;pdb.set_trace()
				#loss.backward()
				#print(self.E_adc_cos)
				delta_cos = -1/math.sqrt(pow(10,self.prec_SQNR_no_vat_cos[i][j]/10))+1/math.sqrt(pow(10,self.prec_SQNR_no_vat_cos[i-1][j]/10))
				delta_sin = -1/math.sqrt(pow(10,self.prec_SQNR_no_vat_sin[i][j]/10))+1/math.sqrt(pow(10,self.prec_SQNR_no_vat_cos[i-1][j]/10))
				print(delta_cos)
				self.E_adc_cos_new[j], self.E_adc_sin_new[j] = E_update( self.E_adc_cos[j],  self.E_adc_sin[j], self.E_adc_cos_new[j], self.E_adc_sin_new[j],i, False, delta_cos, delta_sin)
				#import pdb;pdb.set_trace()
				if i == runs_train -1:
					x_df = pd.DataFrame(self.E_adc_cos_new[j])
					x_df.to_csv('./Data/E_adc_cos.csv', mode='a')
					x_df = pd.DataFrame(self.E_adc_sin_new[j])
					x_df.to_csv('./Data/E_adc_sin.csv', mode='a')
			self.x.append(self.E_adc_cos[6].mean())
			#import pdb;pdb.set_trace()		
			#first_term = False
				#if j ==0:
					#uniform = False
			if basicblock_val_sin[1:fft_size].count_nonzero()!= fft_size-1:
				import pdb;pdb.set_trace()
			#print(basicblock_val.data)
			#import pdb;pdb.set_trace()
			#print(basicblock_no_vat.data)
			i+=1
		plot_curve(self, 6)
		import pdb;pdb.set_trace()
		return self
		
def test(self, ds):
	
	alpha_cos =[]
	alpha_sin=[]
	mul = np.zeros(types)
	basicblock_no_vat_pre_cos=[]
	basicblock_no_vat_pre_sin=[]
	basicblock_no_vat_slice_cos=[]
	basicblock_no_vat_slice_sin=[]
	runs_test = runs
	prec_mean_no_vat_cos = np.zeros((runs_test,types))
	prec_stddev_no_vat_cos = np.zeros((runs_test,types))
	prec_SQNR_no_vat_cos = np.zeros((runs_test,types))
	prec_mean_no_vat_sin = np.zeros((runs_test,types))
	prec_stddev_no_vat_sin = np.zeros((runs_test,types))
	prec_SQNR_no_vat_sin =  np.zeros((runs_test,types))
	E_adc_cos =self.E_adc_cos_new
	E_adc_sin=self.E_adc_sin_new
	indices_cos = self.indices_cos
	indices_sin = self.indices_sin
	weight_cos_quant = self.weight_cos_quant
	E_weight_cos = self.E_weight_cos
	weight_sin_quant = self.weight_sin_quant
	E_weight_sin = self.E_weight_sin
	for k in range(0,types):
		num_row_col = int(fft_size/xbar_size[k]) 
		#E_adc_cos.append(np.zeros((num_row_col, num_row_col)))
		#import pdb;pdb.set_trace()
		#E_adc_sin.append(np.zeros((num_row_col, num_row_col)))
		alpha_cos.append(np.zeros((num_row_col, num_row_col)))
		alpha_sin.append(np.zeros((num_row_col, num_row_col)))
		#map[1,k] = num_row_col
		mul[k] = math.ceil(weight_width/cell_prec[k])
		basicblock_no_vat_pre_cos.append(np.zeros((int(fft_size*mul[k]),types), dtype=np.float64))
		basicblock_no_vat_pre_sin.append(np.zeros((int(fft_size*mul[k]),types), dtype=np.float64))
		basicblock_no_vat_slice_cos.append(np.zeros((int(fft_size*mul[k]),types), dtype=np.float64))
		basicblock_no_vat_slice_sin.append(np.zeros((int(fft_size*mul[k]),types), dtype=np.float64))
	basicblock_no_vat_cos = torch.zeros([fft_size,types], dtype=torch.float64)
	basicblock_no_vat_sin = torch.zeros([fft_size,types], dtype=torch.float64)
	i =0
	"""
	weight_sin_map  = weight_sin
	weight_cos_map  = weight_cos
	weight_sin_map = weight_sin_map[:,indices_sin]
	weight_cos_map = weight_cos_map[:,indices_cos]
	weight_cos_quant, E_weight_cos= weight_quantize(weight_cos, weight_cos_map, cell_prec, weight_width, vat, quant, roll, types)
	weight_sin_quant,E_weight_sin= weight_quantize(weight_sin, weight_sin_map, cell_prec, weight_width, vat, quant, roll, types)
	"""
	for audio,label in tfds.as_numpy(ds):
		#import pdb;pdb.set_trace()
		#audio = np.array(list(audio_dict.items()))[0,1]
		print(type(audio),label,i)
		audio_torch = torch.from_numpy(audio)
		input_fft, E_input, input_fft_unquant,input_fft_slice = input_generate(fft_size, 2, audio_torch.to(torch.float64), in_width,in_split)
		input_fft_cos_up = input_fft_slice
		input_fft_sin_up = input_fft_slice
		basicblock_val_cos, basicblock_val_sin = self.val_generate.forward(input_fft_unquant)
		for j in range(0,types):
			if in_split==0:
				basicblock_no_vat_pre_cos[j], E_adc_cos[j], alpha_cos[j]= conv2d_Q_bit_slice_fn(input_fft_cos_up[0], weight_cos_quant[j], E_weights=E_weight_cos, adc_bits=adc_bits[j], xbar_size=xbar_size[j],  quant = quant, first_term=first_term[j], uniform=uniform[j], E=E_adc_cos[j], alpha = None, w_width=weight_width) 
				basicblock_no_vat_pre_sin[j], E_adc_sin[j], alpha_sin[j] = conv2d_Q_bit_slice_fn(input_fft_sin_up[0], weight_sin_quant[j], E_weights=E_weight_sin, adc_bits=adc_bits[j], xbar_size=xbar_size[j],  quant = quant, first_term=first_term[j], uniform=uniform[j], E=E_adc_sin[j], alpha = None, w_width=weight_width) 
			else:
				for k in range(0, in_width):
					if k == 0:
						basicblock_no_vat_pre_cos[j], E_adc_cos[j], alpha_cos[j]= conv2d_Q_bit_slice_fn(input_fft_cos_up[in_width -1 - k], weight_cos_quant[j], E_weights=E_weight_cos, adc_bits=adc_bits[j], xbar_size=xbar_size[j],  quant = quant, first_term=first_term[j], uniform=uniform[j], E=E_adc_cos[j], alpha = None, w_width=weight_width) 
						basicblock_no_vat_pre_sin[j], E_adc_sin[j], alpha_sin[j] = conv2d_Q_bit_slice_fn(input_fft_sin_up[in_width-1 - k], weight_sin_quant[j], E_weights=E_weight_sin, adc_bits=adc_bits[j], xbar_size=xbar_size[j],  quant = quant, first_term=first_term[j], uniform=uniform[j], E=E_adc_sin[j], alpha = None, w_width=weight_width) 
					else:
						basicblock_no_vat_slice_cos[j], E_adc_cos[j], alpha_cos[j]= conv2d_Q_bit_slice_fn(input_fft_cos_up[in_width -1 - k], weight_cos_quant[j], E_weights=E_weight_cos, adc_bits=adc_bits[j], xbar_size=xbar_size[j],  quant = quant, first_term=first_term[j], uniform=uniform[j], E=E_adc_cos[j], alpha = None, w_width=weight_width) 
						basicblock_no_vat_slice_sin[j], E_adc_sin[j], alpha_sin[j] = conv2d_Q_bit_slice_fn(input_fft_sin_up[in_width-1 - k], weight_sin_quant[j], E_weights=E_weight_sin, adc_bits=adc_bits[j], xbar_size=xbar_size[j],  quant = quant, first_term=first_term[j], uniform=uniform[j], E=E_adc_sin[j], alpha = None, w_width=weight_width) 
						basicblock_no_vat_pre_cos[j] = shift_add_input(basicblock_no_vat_pre_cos[j],basicblock_no_vat_slice_cos[j], k, in_width)
						basicblock_no_vat_pre_sin[j] = shift_add_input(basicblock_no_vat_pre_sin[j],basicblock_no_vat_slice_sin[j], k, in_width)
					if train_mode ==0:
						self=initialize_params_no_train(self)
			if mul[j]>1:
				#print(basicblock_val_cos.data)
				#import pdb;pdb.set_trace()
				#test=torch.range(1,2*(fft_size)-1,2, dtype=int)
				#test1=torch.range(0,2*(fft_size)-2,2, dtype=int)
				#basicblock_no_vat_cos[:,j] = torch.add(basicblock_no_vat_pre_cos[j].index_select(0,test),basicblock_no_vat_pre_cos[j].index_select(0,test1))*E_input
				#basicblock_no_vat_sin[:,j] = torch.add(basicblock_no_vat_pre_sin[j].index_select(0,test),basicblock_no_vat_pre_sin[j].index_select(0,test1))*E_input
				#import pdb;pdb.set_trace()
				basicblock_add= shift_add(basicblock_no_vat_pre_cos[j], weight_width,cell_prec[j])
				basicblock_no_vat_cos[:,j] = (basicblock_add)*E_input
				basicblock_add= shift_add(basicblock_no_vat_pre_sin[j], weight_width,cell_prec[j])
				basicblock_no_vat_sin[:,j] = (basicblock_add)*E_input
			else:
				basicblock_no_vat_cos[:,j] = (basicblock_no_vat_pre_cos[j])*E_input
				basicblock_no_vat_sin[:,j] = (basicblock_no_vat_pre_sin[j])*E_input
			if roll[j] == True:
				basicblock_no_vat_sin[:,j] = basicblock_no_vat_sin[indices_sin.argsort(),j]
				basicblock_no_vat_cos[:,j] = basicblock_no_vat_cos[indices_cos.argsort(),j]
			#if vat[j]:	
			#import pdb;pdb.set_trace()
			"""
			basicblock_no_vat_cos[:,j] = fft_symm(basicblock_no_vat_cos[:,j], True) 
			basicblock_no_vat_sin[:,j] = fft_symm(basicblock_no_vat_sin[:,j], False) 
			"""
			if i >= runs_test-2:
				x_df = pd.DataFrame(E_adc_cos[j])
				x_df.to_csv('./Data/E_adc_cos.csv', mode='a')
				x_df = pd.DataFrame(E_adc_sin[j])
				x_df.to_csv('./Data/E_adc_sin.csv', mode='a')
				x_df = pd.DataFrame(basicblock_val_cos)
				x_df_1 = pd.DataFrame(basicblock_no_vat_cos[:,0])
				if i ==runs_test-2:
					x_df.to_csv('./Data/val_1.csv')
					x_df_1.to_csv('./Data/value_1.csv')
				else:
					x_df.to_csv('./Data/val_2.csv')
					x_df_1.to_csv('./Data/value_2.csv')
				x_df = pd.DataFrame(basicblock_no_vat_cos[:,j])
			prec_mean_no_vat_cos[i][j], prec_stddev_no_vat_cos[i][j], prec_SQNR_no_vat_cos[i][j]= accuracy(basicblock_no_vat_cos.data[1:fft_size,j], basicblock_val_cos.data[1:fft_size], adc_bits)
			prec_mean_no_vat_sin[i][j], prec_stddev_no_vat_sin[i][j], prec_SQNR_no_vat_sin[i][j]= accuracy(basicblock_no_vat_sin.data[1:fft_size,j], basicblock_val_sin.data[1:fft_size], adc_bits)
			if train_mode ==0:
				self=initialize_params_no_train(self)
			#import pdb;pdb.set_trace()
		
		x_df = pd.DataFrame(basicblock_val_sin[331:332])
		x_df.to_csv('./Data/val_core.csv', mode='a')
		x_df = pd.DataFrame(basicblock_val_sin[1024-331:1024-330])
		x_df.to_csv('./Data/val_1_core.csv', mode='a')
		x_df = pd.DataFrame(basicblock_no_vat_sin[331:332,0])
		x_df.to_csv('./Data/value_core.csv', mode='a')
		x_df = pd.DataFrame(basicblock_no_vat_sin[1024-331:1024-330,0])
		x_df.to_csv('./Data/value_1_core.csv', mode='a')
		
						
		#first_term = False
			#if j ==0:
				#uniform = False
		if basicblock_val_sin[1:fft_size].count_nonzero()!= fft_size-1:
			import pdb;pdb.set_trace()
		#print(basicblock_val.data)
		#import pdb;pdb.set_trace()
		#print(basicblock_no_vat.data)
		"""
		if vat:
			basicblock_cos = conv2d_Q_new_fn(input_fft, weight_cos, w_bit=cell_prec, adc_bits=adc_bits, xbar_size=xbar_size, vat = vat, quant = quant) 
			prec_mean_cos[i], prec_stddev_cos[i], prec_SQNR_cos[i]= accuracy(basicblock_cos.data, basicblock_val_cos.data,adc_bits)
			basicblock_sine = conv2d_Q_new_fn(input_fft, weight_sin, w_bit=cell_prec, adc_bits=adc_bits, xbar_size=xbar_size, vat = vat, quant = quant) 
			prec_mean_sin[i], prec_stddev_sin[i], prec_SQNR_sin[i]= accuracy(basicblock_sine.data, basicblock_val_sin.data,adc_bits)
		"""
		
		i+=1	
	import pdb;pdb.set_trace()
	"""
	if vat:
			print ("variations ",vat)
			print("Mean", prec_mean_cos)
			print("Stddev", prec_stddev_cos)
			print("SQNR", prec_SQNR_cos)
			print(" ")
	"""
	#import pdb;pdb.set_trace()
	for j in range(0,types):
		print("crossbar size", xbar_size[j], "roll", roll[j], "cell_prec" ,cell_prec[j], "adc_bits", adc_bits[j], "vat", vat[j])
	#print ("variations False")
		print("cosine")
		print("Mean", prec_mean_no_vat_cos[:,j].mean())
		print("Stddev", prec_stddev_no_vat_cos[:,j].mean())
		print("SQNR", prec_SQNR_no_vat_cos[:,j].mean())
		print(" ")
		print("sine")
		print("Mean", prec_mean_no_vat_sin[:,j].mean())
		print("Stddev", prec_stddev_no_vat_sin[:,j].mean())
		print("SQNR", prec_SQNR_no_vat_sin[:,j].mean())
		print(" ")

def data_from_csv(data,cell_prec,weight_width,xbar_size,fft_size,types):
	E =[]
	E =[]
	mul = np.zeros(types)
	indices = np.array([[3,	5,	7,	12,	15,	17, 20,	23,	28,	33,	42,	51, 68],
			[4,	6,	11,	14,	16,	19,	22,	27,	32, 41,	50,	67, 84]]) 
	data_np=data.to_numpy()
	for k in range(0, types):
		num_row_col = int(fft_size/xbar_size[k])
		mul[k] = math.ceil(weight_width/cell_prec[k]) 
		#import pdb;pdb.set_trace()
		E.append(data_np[indices[0][k]:indices[1][k],1:(int((indices[1][k]-indices[0][k])*mul[k])+1)]*1.1)
	#import pdb;pdb.set_trace()
	return E

class no_train():
	def __init__(self):
		#import pdb;pdb.set_trace()
		super(no_train, self).__init__()
		weight_cos, weight_sin = weight_generate_main(fft_size)
		self.weight_cos = weight_cos
		self.weight_sin = weight_sin
		self.val_generate = val_generate(weight_cos, weight_sin)
		#import pdb;pdb.set_trace()
		self = initialize_params_no_train(self)
			
	def forward(self, ds):
		"""
		i=0
		for audio,label in tfds.as_numpy(ds):
			#import pdb;pdb.set_trace()
			#audio = np.array(list(audio_dict.items()))[0,1]
			print(type(audio),label,i)
			audio_torch = torch.from_numpy(audio)
			in_type = 2
			input_src = audio_torch.to(torch.float64)
			#import pdb;pdb.set_trace()
			input_fft, E_input, input_fft_unquant, input_fft_slice= input_generate(fft_size, in_type, input_src, in_width, in_split)
			input_fft_cos_up = input_fft_slice
			input_fft_sin_up = input_fft_slice
			basicblock_val_cos,basicblock_val_sin=self.val_generate.forward(input_fft_unquant)	
			if i ==0:
				self = weight_quantize_main(self, basicblock_val_cos, basicblock_val_sin) 
			i+=1
		"""
		return self
		

class train_pattern():
	def __init__(self):
		#import pdb;pdb.set_trace()
		super(train_pattern, self).__init__()
		weight_cos, weight_sin = weight_generate_main(fft_size)
		self.weight_cos = weight_cos
		self.weight_sin = weight_sin
		self.val_generate = val_generate(weight_cos, weight_sin)
		self=initialize_params(self, runs_train)
		self.conv_main = conv_main(self.weight_cos,self.weight_sin)
		self.conv_grad=conv_grad(self.mul, self.prec_mean_no_vat_cos, self.prec_mean_no_vat_sin, self.prec_stddev_no_vat_cos, self.prec_stddev_no_vat_sin,  self.prec_SQNR_no_vat_cos,  self.prec_SQNR_no_vat_sin)
		self.x=[]
		#import pdb;pdb.set_trace()
	
	def forward(self, ds):
		i=0
		k=0
		#offset_cos = offset(weight_cos, in_width, adc_bits, in_split)
		#offset_sin = offset(weight_sin, in_width, adc_bits, in_split)
		for audio,label in tfds.as_numpy(ds):
			#import pdb;pdb.set_trace()
			#audio=example['audio']
			#audio = np.array(list(audio_dict.items()))[0,1]
			print(type(audio),label,i)
		#for k in range(ds.shape[0]):
			#audio =ds.numpy()[i]
			audio_torch = torch.from_numpy(audio)
			in_type = 2
			input_src = audio_torch.to(torch.float64)
			#import pdb;pdb.set_trace()
			input_fft, E_input, input_fft_unquant, input_fft_slice= input_generate(fft_size, in_type, input_src, in_width, in_split)
			input_fft_cos_up = input_fft_slice
			input_fft_sin_up = input_fft_slice
			basicblock_val_cos,basicblock_val_sin=self.val_generate.forward(input_fft_unquant)	
			if i ==0:
				self = weight_quantize_main(self, basicblock_val_cos, basicblock_val_sin) 
			#import pdb;pdb.set_trace()
			self.sign_prob =prob_sign(input_fft)
			for i in range(0,runs_train):
				for j in range(0,types):
					if i ==0:
						self.conv_main.sync(self.E_adc_cos_new, self.E_adc_sin_new, self.E_adc_cos,  self.E_adc_sin, self.sign_prob,self.weight_cos_quant, self.weight_sin_quant, self.alpha_cos, self.alpha_sin, self.E_weight_cos, self.E_weight_sin, self.E_adc_cos_slice, self.E_adc_sin_slice, self.basicblock_no_vat_pre_cos, self.basicblock_no_vat_pre_sin)
						self.conv_main.forward(input_fft_cos_up, input_fft_sin_up, j, False)
						self.E_adc_cos_new[j], self.E_adc_sin_new[j] = E_update( self.E_adc_cos[j],  self.E_adc_sin[j], self.E_adc_cos_new[j], self.E_adc_sin_new[j],i, True, self.prec_mean_no_vat_cos[i-1][j], self.prec_mean_no_vat_sin[i-1][j])
					#import pdb;pdb.set_trace()
					#print(basicblock_no_vat_slice_cos[0].unique())
					else:
						self.conv_main.sync(self.E_adc_cos_new, self.E_adc_sin_new, self.E_adc_cos,  self.E_adc_sin, self.sign_prob,self.weight_cos_quant, self.weight_sin_quant, self.alpha_cos, self.alpha_sin, self.E_weight_cos, self.E_weight_sin, self.E_adc_cos_slice, self.E_adc_sin_slice, self.basicblock_no_vat_pre_cos, self.basicblock_no_vat_pre_sin)
						self.conv_main.forward(input_fft_cos_up, input_fft_sin_up, j, True)
					self.conv_grad.sync(self.basicblock_no_vat_pre_cos, self.basicblock_no_vat_pre_sin, self.indices_sin, self.indices_cos, self.basicblock_no_vat_cos, self.basicblock_no_vat_sin)
					self.conv_grad.forward(i,j,E_input, basicblock_val_cos, basicblock_val_sin)
					self.E_adc_cos_new[j] = self.E_adc_cos[j]-0.05
					self.E_adc_sin_new[j] = self.E_adc_sin[j]-0.05
					#import pdb;pdb.set_trace()
					if i == runs_train -1:
						x_df = pd.DataFrame(self.E_adc_cos_new[j])
						x_df.to_csv('./Data/E_adc_cos.csv', mode='a')
						x_df = pd.DataFrame(self.E_adc_sin_new[j])
						x_df.to_csv('./Data/E_adc_sin.csv', mode='a')
				print(self.E_adc_cos_new[6].mean(), i)
				#import pdb;pdb.set_trace()		
				self.x.append(self.E_adc_cos[6].mean())
				i+=1
			#import pdb;pdb.set_trace()		
			#first_term = False
				#if j ==0:
					#uniform = False
			if basicblock_val_sin[1:fft_size].count_nonzero()!= fft_size-1:
				import pdb;pdb.set_trace()
			#print(basicblock_val.data)
			#import pdb;pdb.set_trace()
			#print(basicblock_no_vat.data)
		plot_curve(self,6)
		import pdb;pdb.set_trace()
		return self
	

class train_prec():
	def __init__(self):
		#import pdb;pdb.set_trace()
		super(train_prec, self).__init__()
		weight_cos, weight_sin = weight_generate_main(fft_size)
		self.weight_cos = weight_cos
		self.weight_sin = weight_sin
		self.val_generate = val_generate(weight_cos, weight_sin)
		self=initialize_params(self, runs_train)
		self.conv_main = conv_main(self.weight_cos,self.weight_sin)
		self.conv_grad=conv_grad(self.mul, self.prec_mean_no_vat_cos, self.prec_mean_no_vat_sin, self.prec_stddev_no_vat_cos, self.prec_stddev_no_vat_sin,  self.prec_SQNR_no_vat_cos,  self.prec_SQNR_no_vat_sin)
		self.x=[]
		#import pdb;pdb.set_trace()
	
	def forward(self, ds):
		i=0
		#offset_cos = offset(weight_cos, in_width, adc_bits, in_split)
		#offset_sin = offset(weight_sin, in_width, adc_bits, in_split)
		#for example in tfds.as_numpy(ds):
			#import pdb;pdb.set_trace()
			#audio=example['audio']
			#audio = np.array(list(audio_dict.items()))[0,1]
		#for i in range(ds.shape[0]):
		for audio,label in tfds.as_numpy(ds):
			#audio =ds.numpy()[i]
			audio_torch = torch.from_numpy(audio)
			#print(type(audio),label,i)
			in_type = 2
			input_src = audio_torch.to(torch.float64)
			#import pdb;pdb.set_trace()
			input_fft, E_input, input_fft_unquant, input_fft_slice= input_generate(fft_size, in_type, input_src, in_width, in_split)
			input_fft_cos_up = input_fft_slice
			input_fft_sin_up = input_fft_slice
			basicblock_val_cos,basicblock_val_sin=self.val_generate.forward(input_fft_unquant)	
			#import pdb;pdb.set_trace()
			if i ==0:
				self = weight_quantize_main(self, basicblock_val_cos, basicblock_val_sin) 
			
			#import pdb;pdb.set_trace()
			self.sign_prob =prob_sign(input_fft)
			
			for j in range(0,types):
				#import pdb;pdb.set_trace()
				if input_fft_cos_up[j].unique().size()[0]>2**in_width:
					import pdb;pdb.set_trace()
				if self.weight_cos_quant[j].unique().size()[0]>2**(cell_prec[j])*3/2:
					import pdb;pdb.set_trace()
				if i ==0:
					self.conv_main.sync(self.E_adc_cos_new, self.E_adc_sin_new, self.E_adc_cos,  self.E_adc_sin, self.sign_prob,self.weight_cos_quant, self.weight_sin_quant, self.alpha_cos, self.alpha_sin, self.E_weight_cos, self.E_weight_sin, self.E_adc_cos_slice, self.E_adc_sin_slice, self.basicblock_no_vat_pre_cos, self.basicblock_no_vat_pre_sin)
					self.conv_main.forward(input_fft_cos_up, input_fft_sin_up, j, False)
					self.E_adc_cos_new[j], self.E_adc_sin_new[j] = E_update( self.E_adc_cos[j],  self.E_adc_sin[j], self.E_adc_cos_new[j], self.E_adc_sin_new[j],i, True, self.prec_mean_no_vat_cos[i-1][j], self.prec_mean_no_vat_sin[i-1][j])
					#import pdb;pdb.set_trace()
				#print(basicblock_no_vat_slice_cos[0].unique())
				else:
					self.conv_main.sync(self.E_adc_cos_new, self.E_adc_sin_new, self.E_adc_cos,  self.E_adc_sin, self.sign_prob,self.weight_cos_quant, self.weight_sin_quant, self.alpha_cos, self.alpha_sin, self.E_weight_cos, self.E_weight_sin, self.E_adc_cos_slice, self.E_adc_sin_slice, self.basicblock_no_vat_pre_cos, self.basicblock_no_vat_pre_sin)
					self.conv_main.forward(input_fft_cos_up, input_fft_sin_up, j, True)
				self.conv_grad.sync(self.basicblock_no_vat_pre_cos, self.basicblock_no_vat_pre_sin, self.indices_sin, self.indices_cos, self.basicblock_no_vat_cos, self.basicblock_no_vat_sin)
				self.conv_grad.forward(i,j,E_input, basicblock_val_cos, basicblock_val_sin)
				#self.E_adc_cos_new[j] = self.E_adc_cos[j]-0.05
				#self.E_adc_sin_new[j] = self.E_adc_sin[j]-0.05
				#import pdb;pdb.set_trace()
				if i == runs_train -1:
					x_df = pd.DataFrame(self.E_adc_cos_new[j])
					x_df.to_csv('./Data/E_adc_cos.csv', mode='a')
					x_df = pd.DataFrame(self.E_adc_sin_new[j])
					x_df.to_csv('./Data/E_adc_sin.csv', mode='a')
			#print(self.E_adc_cos_new[6].mean(), i)
			#import pdb;pdb.set_trace()		
			i+=1
			#import pdb;pdb.set_trace()		
			#first_term = False
				#if j ==0:
					#uniform = False
			if basicblock_val_sin[1:fft_size].count_nonzero()!= fft_size-1:
				import pdb;pdb.set_trace()
			#print(basicblock_val.data)
			#import pdb;pdb.set_trace()
			#print(basicblock_no_vat.data)
		plot_curve_prec(self)
		import pdb;pdb.set_trace()
		return self
	
class test_new():
	def __init__(self, adc_quant_factor, E_adc_cos, E_adc_sin, E_input):
		#import pdb;pdb.set_trace()
		super(test_new, self).__init__()
		weight_cos, weight_sin = weight_generate_main(fft_size)
		self.weight_cos = weight_cos
		self.weight_sin = weight_sin
		self.val_generate = val_generate(self.weight_cos, self.weight_sin)
		self=initialize_params_test(self, runs_test, E_adc_cos, E_adc_sin)
		self.conv_main = conv_main(self.weight_cos,self.weight_sin)
		self.conv_grad=conv_grad(self.mul, self.prec_mean_no_vat_cos, self.prec_mean_no_vat_sin, self.prec_stddev_no_vat_cos, self.prec_stddev_no_vat_sin,  self.prec_SQNR_no_vat_cos,  self.prec_SQNR_no_vat_sin, self.prec_MAE_no_vat_cos, self.prec_MAE_no_vat_sin)
		self.adc_quant_factor=adc_quant_factor
		self.E_input=None
		#import pdb;pdb.set_trace()
	
	def forward(self, ds):
		i=0
		#offset_cos = offset(weight_cos, in_width, adc_bits, in_split)
		#offset_sin = offset(weight_sin, in_width, adc_bits, in_split)
		#for example in tfds.as_numpy(ds):
			#import pdb;pdb.set_trace()
			#audio=example['audio']
			#audio = np.array(list(audio_dict.items()))[0,1]
		#for i in range(ds.shape[0]):
		for audio,label in tfds.as_numpy(ds):
			#audio =ds.numpy()[i]
			audio_torch = torch.from_numpy(audio)
			#print(type(audio),label,i)
			in_type = 2
			input_src = audio_torch.to(torch.float64)
			#import pdb;pdb.set_trace()
			input_fft, E_input, input_fft_unquant, input_fft_slice,max_fft= input_generate(fft_size, in_type, input_src, in_width, in_split, self.adc_quant_factor, self.E_input)
			input_fft_cos_up = input_fft_slice
			input_fft_sin_up = input_fft_slice
			basicblock_val_cos,basicblock_val_sin=self.val_generate.forward(input_fft_unquant)	
			#import pdb;pdb.set_trace()
			if i ==0:
				self = weight_quantize_main(self, basicblock_val_cos, basicblock_val_sin) 
			
			#import pdb;pdb.set_trace()
			self.sign_prob =prob_sign(input_fft)
			#import pdb;pdb.set_trace()
			
			for j in range(0,types):
				#import pdb;pdb.set_trace()
				if in_quant:
					if input_fft_cos_up[j].unique().size()[0]>2**in_width:
						import pdb;pdb.set_trace()
				if wg_quant:
					if not vat[j]:
						if self.weight_cos_quant[j].unique().size()[0]>2**(cell_prec[j])*3/2:
							import pdb;pdb.set_trace()
				#import pdb;pdb.set_trace()
				if i ==0:
					self.conv_main.sync(self.E_adc_cos_new, self.E_adc_sin_new, self.E_adc_cos,  self.E_adc_sin, self.sign_prob,self.weight_cos_quant, self.weight_sin_quant, self.alpha_cos, self.alpha_sin, self.E_weight_cos, self.E_weight_sin, self.E_adc_cos_slice, self.E_adc_sin_slice, self.basicblock_no_vat_pre_cos, self.basicblock_no_vat_pre_sin)
					#import pdb;pdb.set_trace()
					self.conv_main.forward(input_fft_cos_up, input_fft_sin_up, j, True, max_fft)
					#self.E_adc_cos_new[j], self.E_adc_sin_new[j] = E_update( self.E_adc_cos[j],  self.E_adc_sin[j], self.E_adc_cos_new[j], self.E_adc_sin_new[j],i, True, self.prec_mean_no_vat_cos[i-1][j], self.prec_mean_no_vat_sin[i-1][j])
					#import pdb;pdb.set_trace()
				#print(basicblock_no_vat_slice_cos[0].unique())
				else:
					self.conv_main.sync(self.E_adc_cos_new, self.E_adc_sin_new, self.E_adc_cos,  self.E_adc_sin, self.sign_prob,self.weight_cos_quant, self.weight_sin_quant, self.alpha_cos, self.alpha_sin, self.E_weight_cos, self.E_weight_sin, self.E_adc_cos_slice, self.E_adc_sin_slice, self.basicblock_no_vat_pre_cos, self.basicblock_no_vat_pre_sin)
					self.conv_main.forward(input_fft_cos_up, input_fft_sin_up, j, True, max_fft)
				#import pdb;pdb.set_trace()
				self.conv_grad.sync(self.basicblock_no_vat_pre_cos, self.basicblock_no_vat_pre_sin, self.indices_sin, self.indices_cos, self.basicblock_no_vat_cos, self.basicblock_no_vat_sin)
				self.conv_grad.forward(i,j,E_input, basicblock_val_cos, basicblock_val_sin)
				#self.E_adc_cos_new[j] = self.E_adc_cos[j]-0.05
				#self.E_adc_sin_new[j] = self.E_adc_sin[j]-0.05
				#import pdb;pdb.set_trace()
				if i == runs_train -1:
					x_df = pd.DataFrame(self.E_adc_cos_new[j])
					x_df.to_csv('./Data/E_adc_cos.csv', mode='a')
					x_df = pd.DataFrame(self.E_adc_sin_new[j])
					x_df.to_csv('./Data/E_adc_sin.csv', mode='a')
			#print(self.E_adc_cos_new[6].mean(), i)
			#import pdb;pdb.set_trace()		
			i+=1
			#import pdb;pdb.set_trace()		
			#first_term = False
				#if j ==0:
					#uniform = False
			if basicblock_val_sin[1:fft_size].count_nonzero()!= fft_size-1:
				import pdb;pdb.set_trace()
			#print(basicblock_val.data)
			#import pdb;pdb.set_trace()
			#print(basicblock_no_vat.data)
		#plot_curve_prec(self)
		SQNR=[]
		MAE=[]
		for j in range(0,types):
			print("crossbar size", xbar_size[j], "roll", roll[j], "cell_prec" ,cell_prec[j], "adc_bits", adc_bits[j], "vat", vat[j])
		#print ("variations False")
			print("cosine")
			print("Mean", self.prec_mean_no_vat_cos[:,j].mean())
			print("Stddev", self.prec_stddev_no_vat_cos[:,j].mean())
			print("SQNR", self.prec_SQNR_no_vat_cos[:,j].mean())
			#import pdb;pdb.set_trace()
			print("MAE", self.prec_MAE_no_vat_cos[:,j].mean())
			print(" ")
			print("sine")
			print("Mean", self.prec_mean_no_vat_sin[:,j].mean())
			print("Stddev", self.prec_stddev_no_vat_sin[:,j].mean())
			print("SQNR", self.prec_SQNR_no_vat_sin[:,j].mean())
			print("MAE", self.prec_MAE_no_vat_sin[:,j].mean())
			print(" ")
		#import pdb;pdb.set_trace()
			SQNR.append(self.prec_SQNR_no_vat_cos[:,j].mean())
			MAE.append(self.prec_MAE_no_vat_sin[:,j].mean())
		return self, SQNR, MAE

class train_factor():
	def __init__(self):
		#import pdb;pdb.set_trace()
		super(train_factor, self).__init__()
		weight_cos, weight_sin = weight_generate_main(fft_size)
		self.weight_cos = weight_cos
		self.weight_sin = weight_sin
		self.val_generate = val_generate(weight_cos, weight_sin)
		self=initialize_params(self, runs_train)
		self.adc_quant_factor=1
		self.E_input=0
		#import pdb;pdb.set_trace()
	
	def forward(self, ds):
		i=0
		k=0
		adc_quant_factor = torch.zeros([runs_train])
		#offset_cos = offset(weight_cos, in_width, adc_bits, in_split)
		#offset_sin = offset(weight_sin, in_width, adc_bits, in_split)
		for audio,label in tfds.as_numpy(ds):
			#import pdb;pdb.set_trace()
			#audio=example['audio']
			#audio = np.array(list(audio_dict.items()))[0,1]
			print(type(audio),label,i)
		#for k in range(ds.shape[0]):
			#audio =ds.numpy()[i]
			audio_torch = torch.from_numpy(audio)
			in_type = 2
			input_src = audio_torch.to(torch.float64)
			#import pdb;pdb.set_trace()
			input_fft, E_input, input_fft_unquant, input_fft_slice, max_fft= input_generate(fft_size, in_type, input_src, in_width, in_split, self.adc_quant_factor, None)
			basicblock_val_cos,basicblock_val_sin=self.val_generate.forward(input_fft_unquant)	
			if i ==0:
				self = weight_quantize_main(self, basicblock_val_cos, basicblock_val_sin) 
			#import pdb;pdb.set_trace()
			self.sign_prob =prob_sign(input_fft)
			adc_quant_factor[i]=basicblock_val_sin.abs().max()/(max_fft*E_input)
			#import pdb;pdb.set_trace()
			self.E_input=max(self.E_input, E_input)
			x_df = pd.DataFrame(np.array([E_input]))
			x_df.to_csv('./Data/E_input.csv', mode='a')
			#import pdb;pdb.set_trace()		
			i+=1
			#first_term = False
				#if j ==0:
					#uniform = False
			if basicblock_val_sin[1:fft_size].count_nonzero()!= fft_size-1:
				import pdb;pdb.set_trace()
			#print(basicblock_val.data)
			#import pdb;pdb.set_trace()
			#print(basicblock_no_vat.data)
		self.adc_quant_factor=adc_quant_factor.quantile(0.2).item()
		"""
		for j in range(0, types):
			k=0
			incr=int(xbar_size[j]/weight_width*cell_prec[j])
			for col in range(0, fft_size, incr):
				#import pdb;pdb.set_trace()
				self.E_adc_cos[j][0][k] = basicblock_val_cos[col:col+incr].abs().max()/(max_fft*E_input*self.adc_quant_factor)
				self.E_adc_sin[j][0][k] = basicblock_val_sin[col:col+incr].abs().max()/(max_fft*E_input*self.adc_quant_factor)
				k+=1
			rows=int(fft_size/xbar_size[j])
			self.E_adc_cos_new[j]=np.repeat(self.E_adc_cos[j],rows,axis=0)[0:rows]
			self.E_adc_sin_new[j]=np.repeat(self.E_adc_sin[j],rows,axis=0)[0:rows]
		"""
		#import pdb;pdb.set_trace()
		return self
"""
for j in range(0, types):
		k=0
		incr=int(xbar_size[j]/weight_width*cell_prec[j])
		for col in range(0, fft_size, incr):
			#import pdb;pdb.set_trace()
			self.E_adc_cos[j][0][k] = basicblock_val_cos[col:col+incr].abs().max()/(max_fft*E_input*adc_quant_factor[i])
			self.E_adc_sin[j][0][k] = basicblock_val_sin[col:col+incr].abs().max()/(max_fft*E_input*adc_quant_factor[i])
			k+=1
		rows=int(fft_size/xbar_size[j])
		self.E_adc_cos_new[j]=np.repeat(self.E_adc_cos[j],rows,axis=0)[0:rows]
		self.E_adc_sin_new[j]=np.repeat(self.E_adc_sin[j],rows,axis=0)[0:rows]
	x_df = pd.DataFrame(self.E_adc_cos_new[2])
	x_df.to_csv('./Data/E_adc_cos_new.csv', mode='a')
"""