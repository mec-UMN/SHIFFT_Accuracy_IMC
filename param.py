import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser(description='Training network for image classification',
								 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--vat', dest='vat', action='store_true',
					help='Add device variations, SAF to the model')
parser.add_argument('--in_width', type=int, default=2,
					help='Input Width')
parser.add_argument('--weight_width', type=int, default=2,
					help='Weights data width')
parser.add_argument('--cell_prec', type=int, default=2,
					help='Weights Quantization precision')
parser.add_argument('--adc_bits', type=int, default=3,
					help='ADC precision for HW aware training')
parser.add_argument('--xbar_size', type=int, default=64,
					help='Number of rows of the crossbar')
parser.add_argument('--fft_size', type=int, default=1024,
					help='Size of FFT network')
parser.add_argument('--pattern', type=int, default=1,
					help='input pattern')
parser.add_argument('--quant', dest='quant', action='store_true',
					help='Add quantization to the model')
parser.add_argument('--runs', type=int, default=1,
					help='Number of runs')
parser.add_argument('--data_path', default='./dataset/',
					type=str, help='Path to dataset')
parser.add_argument('--manualSeed', type=int, default=5000, help='manual seed')
parser.add_argument('--dataset', type=str, choices=['crema_d', 'spoken_digit', 'speech_commands','groove', 'nsynth', 'ecg'],
					help='Choose dataset')
parser.add_argument('--pre_trained_adc_quant', dest='pre_trained_adc_quant', action='store_true',
					help='Load Pre_trained ADC Quant')


# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--gpu_id', type=int, default=1,
					help='device range [0,ngpu-1]')
parser.add_argument('--workers', type=int, default=4,
					help='number of data loading workers (default: 2)')
                    
args = parser.parse_args()
dataset = args.dataset
#xbar_size = args.xbar_size
types = 2
train_mode =4
if train_mode ==0 or train_mode ==3 or train_mode==4:
	xbar_size= np.repeat(np.array([1024]), types)
	#xbar_size = np.repeat(np.array([1024, 512, 256, 128, 64, 64]),1)
	#xbar_size = np.repeat(np.array([4096, 2048, 1024, 512, 256, 128]),1)
	#xbar_size = np.repeat(np.array([256, 128, 64, 64, 64, 64]),1)
	adc_bits=np.repeat(np.array([6]), types)
	#adc_bits=np.array([8, 7, 6, 5, 4, 3])
else:
	#xbar_size = [256, 256, 64, 128,256,128,128, 64, 64, 128, 128, 256, 256]
	#xbar_size= np.repeat(np.array([128]), types)
	xbar_size = np.repeat(np.array([1024, 512, 1024, 512, 1024, 512, 256, 128, 64, 64]),1)
	adc_bits= np.repeat(np.array([6, 6, 6, 6, 6]), 2)
first_term = np.repeat(np.array([True],dtype=bool), types)
#uniform =np.array([True, True, False, False, False, False, False, False],dtype=bool)
uniform =np.repeat(np.array([False], dtype=bool),types)
roll = np.repeat(np.array([False],dtype=bool),types)
map = np.zeros((2,types))
"""
xbar_size =[512, 512]
map = np.zeros((2,types)),
adc_bits= [5,5]
first_term = np.array([False, False],dtype=bool)
uniform = np.array([False, False],dtype=bool)
"""
#cell_prec = args.cell_prec
cell_prec= np.repeat(np.array([args.cell_prec]),types)
#cell_prec= np.repeat(np.array([2,4]),1)
vat = np.tile(np.array([False, args.vat]), int(types/2))
#vat = args.vat
fft_size = args.fft_size
pattern = args.pattern
quant = args.quant
runs_test = args.runs
in_width = args.in_width
weight_width = args.weight_width
pre_trained_adc_quant=args.pre_trained_adc_quant
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if args.ngpu == 1:
	# make only device #gpu_id visible, then
	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()  # check 

# Give a random seed if no manual configuration
if args.manualSeed is None:
	args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
np.random.seed(args.manualSeed)

if args.use_cuda:
	torch.cuda.manual_seed_all(args.manualSeed)

cudnn.benchmark = True
in_split =0
rate = 0
runs_train =20
if train_mode==0:
	runs_train=1
elif train_mode==1:
	runs_train =20
elif train_mode==2:
	runs_train=30
elif train_mode==4:
	runs_train=400
	

if quant:
	in_quant=1
	wg_quant=1
	adc_quant=0
else:
	in_quant=0
	wg_quant=0
	adc_quant=0