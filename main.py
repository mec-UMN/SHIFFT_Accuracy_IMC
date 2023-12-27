import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
import sys
import shutil
import time
import pandas as pd
import datetime

from fft import conv2d_Q_new_fn, conv2d_Q_bit_slice_fn,conv2d_Q_val_fn
from model import model_test, test, model_train, train, data_from_csv, no_train, train_pattern, train_prec, test_new, train_factor
from Utils import  input_generate, weight_generate, accuracy, weight_generate_imc, shift_add, map_update
from param import types,fft_size,xbar_size, uniform, weight_width, in_width, cell_prec,adc_bits, roll, quant, first_term, in_split,vat, runs_test, runs_train,args, pre_trained_adc_quant, dataset, train_mode

def main():
		
	if not os.path.isdir(args.data_path):
		os.makedirs(args.data_path)
	#import pdb;pdb.set_trace()
	
	if not pre_trained_adc_quant:
		ds = model_train(dataset, args, 1)
		#if train_mode ==0:
			#net=no_train() 
			#net=net.forward(ds)
		if train_mode ==1:
			ds = model_train(dataset, args, runs_train)
			net=train() 
			net=net.forward(ds=ds)
		elif train_mode ==2:
			net = train_pattern()
			net = net.forward(ds)
		elif train_mode ==3:
			net = train_prec()
			net = net.forward(ds)
		elif train_mode ==4:
			ds = model_train(dataset, args, runs_train)
			net = train_factor()
			net = net.forward(ds)
	else:
		data=pd.read_csv('./Data/E_adc_cos.csv')
		E_adc_cos=data_from_csv(data,cell_prec,weight_width,xbar_size,fft_size,types)
		data=pd.read_csv('./Data/E_adc_sin.csv')
		E_adc_sin=data_from_csv(data,cell_prec,weight_width,xbar_size,fft_size,types)
		#import pdb;pdb.set_trace()
		indices_cos=None
		indices_sin=None
	ds = model_test(dataset, args)
	if train_mode==0:
		net=test_new(1)
		net=net.forward(ds)
	elif train_mode==4:
		net=test_new(net.adc_quant_factor, net.E_adc_cos_new, net.E_adc_sin_new, net.E_input)
		net,SQNR=net.forward(ds)
	else:
		test(net, ds)
	
	# Get the current date and time
	current_time = datetime.datetime.now()

	# Create a formatted string for the timestamp
	timestamp_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")

	# Combine the timestamp with a file extension or any other desired format
	file_name = f"./Outputs/output_{timestamp_str}.csv"
	x_df=pd.DataFrame(SQNR)
	x_df.to_csv(file_name,  index=False, header=False)
	
if __name__ == '__main__':
	main()