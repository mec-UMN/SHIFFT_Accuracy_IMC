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

def add_vat(w_bit, weight_q):
    shape_w = weight_q.size()
    weight_pre_vat=weight_q
    #import pdb;pdb.set_trace()
    if w_bit==1:
        var_exp_1 =  torch.exp(torch.empty(shape_w).normal_(0, 0.1035))
        weight_q = weight_q * var_exp_1

    elif w_bit==2:
        #lsb
        weight_2bit_1_1=torch.clone(weight_q)
        #print_log(unique,log)
        weight_2bit_1_1[weight_2bit_1_1.abs()>0.4]=0
        
        weight_2bit_1=torch.clone(weight_2bit_1_1)
        #if self.iter==0:
        #    self.var_exp_2_1 = torch.exp(torch.empty(shape_w).normal_(0, 0.2760))
        var_exp_2_1 = torch.exp(torch.empty(shape_w).normal_(0, 0.276))
        weight_2bit_vat1=torch.clone(weight_2bit_1*var_exp_2_1)
        #print_log(torch.linalg.norm(weight_2bit_vat1),log)
        
        #MSB
        weight_2bit_1_2=torch.clone(weight_q)
        weight_2bit_1_2[weight_2bit_1_2.abs()<0.9]=0
        weight_2bit_2=torch.clone(weight_2bit_1_2)
        #if self.iter==0:
        #   self.var_exp_2_2 = torch.exp(torch.empty(shape_w).normal_(0, 0.1035))
        var_exp_2_2 = torch.exp(torch.empty(shape_w).normal_(0, 0.1035))
        weight_2bit_vat2=torch.clone(weight_2bit_2*var_exp_2_2)
        #import pdb;pdb.set_trace()
        weight_q=torch.add(weight_2bit_vat1,weight_2bit_vat2)

        #import pdb;pdb.set_trace()
        #	
    elif w_bit ==3:

        weight_3bit_1_1=torch.clone(weight_q)
        weight_3bit_1_1[weight_3bit_1_1.abs()>0.143]=0
        weight_3bit_1=torch.clone(weight_3bit_1_1)
        #if self.iter==0:
        #	self.var_exp_3_1 = torch.exp(torch.empty(shape_w).normal_(0, 0.3549))
        var_exp_3_1 = torch.exp(torch.empty(shape_w).normal_(0, 0.3549))
        #print(var_exp_3_1.norm())
        weight_3bit_vat1=torch.clone(weight_3bit_1*var_exp_3_1)

        
        #-----------------------2nd bit--------------


        weight_3bit_1_2=torch.clone(weight_q)
        weight_3bit_1_2[weight_3bit_1_2.abs()<0.42]=0
        weight_3bit_1_2[weight_3bit_1_2.abs()>0.43]=0
        
        weight_3bit_2=torch.clone(weight_3bit_1_2)
        #if self.iter==0:

        #	self.var_exp_3_2 = torch.exp(torch.empty(shape_w).normal_(0, 0.2259))
        var_exp_3_2 = torch.exp(torch.empty(shape_w).normal_(0, 0.2259))
        weight_3bit_vat2=torch.clone(weight_3bit_2*var_exp_3_2)

        #-----------------------3rd bit------------------------
        weight_3bit_1_3=torch.clone(weight_q)
        weight_3bit_1_3[weight_3bit_1_3.abs()<0.7]=0
        #import pdb;pdb.set_trace()
        weight_3bit_3=torch.clone(weight_3bit_1_3)

        #if self.iter==0:
        #	self.var_exp_3_3 = torch.exp(torch.empty(shape_w).normal_(0, 0.1898))
        var_exp_3_3 = torch.exp(torch.empty(shape_w).normal_(0, 0.1898))
        weight_3bit_vat3=torch.clone(weight_3bit_3*var_exp_3_3)
        #print(weight_3bit_vat3)

        weight_3bit=torch.add(weight_3bit_vat1,weight_3bit_vat2)
        weight_3bit=torch.add(weight_3bit,weight_3bit_vat3)

        weight_q=weight_3bit
    else:

        weight_4bit_1_1=torch.clone(weight_q)
        weight_4bit_1_1[weight_4bit_1_1.abs()>0.07]=0
        weight_4bit_1=torch.clone(weight_4bit_1_1)
        #if self.iter==0:
        #	self.var_exp_4_1 = torch.exp(torch.empty(shape_w).normal_(0, 0.3549))
        var_exp_4_1 = torch.exp(torch.empty(shape_w).normal_(0, 0.354929))
        #print(var_exp_4_1.norm())
        weight_4bit_vat1=torch.clone(weight_4bit_1*var_exp_4_1)

        
        #-----------------------2nd bit--------------


        weight_4bit_1_2=torch.clone(weight_q)
        weight_4bit_1_2[weight_4bit_1_2.abs()<0.19]=0
        weight_4bit_1_2[weight_4bit_1_2.abs()>0.21]=0
        
        weight_4bit_2=torch.clone(weight_4bit_1_2)

        var_exp_4_2 = torch.exp(torch.empty(shape_w).normal_(0, 0.2761))
        weight_4bit_vat2=torch.clone(weight_4bit_2*var_exp_4_2)

        #-----------------------3rd bit------------------------
        weight_4bit_1_3=torch.clone(weight_q)
        weight_4bit_1_3[weight_4bit_1_3.abs()<0.3]=0
        weight_4bit_1_3[weight_4bit_1_3.abs()>0.47]=0
        #import pdb;pdb.set_trace()
        weight_4bit_3=torch.clone(weight_4bit_1_3)

        var_exp_4_3 = torch.exp(torch.empty(shape_w).normal_(0, 0.0836))
        weight_4bit_vat3=torch.clone(weight_4bit_3*var_exp_4_3)
        #print(weight_4bit_vat3)

        #-----------------------4th bit------------------------
        weight_4bit_1_4=torch.clone(weight_q)
        weight_4bit_1_4[weight_4bit_1_4.abs()<0.6]=0
        #import pdb;pdb.set_trace()
        weight_4bit_4=torch.clone(weight_4bit_1_4)

        var_exp_4_4 = torch.exp(torch.empty(shape_w).normal_(0, 0.0560))
        weight_4bit_vat4=torch.clone(weight_4bit_4*var_exp_4_4)
        #print(weight_4bit_vat3)

        weight_4bit=torch.add(weight_4bit_vat1,weight_4bit_vat2)
        weight_4bit=torch.add(weight_4bit,weight_4bit_vat3)
        weight_4bit=torch.add(weight_4bit,weight_4bit_vat4)
        weight_q=weight_4bit
    
    #import pdb;pdb.set_trace()
    
    return weight_q

def add_vat_new(w_bit, weight_q):
    shape_w = weight_q.size()
    weight_pre_vat=weight_q
    #import pdb;pdb.set_trace()
    if w_bit==1:
        var_exp_1 =  torch.exp(torch.empty(shape_w).normal_(0, 0.1035))
        weight_q = weight_q * var_exp_1

    elif w_bit==2:
        #lsb
        weight_2bit_1_1=torch.clone(weight_q)
        #print_log(unique,log)
        weight_2bit_1_1[weight_2bit_1_1.abs()>0.4]=0
        
        weight_2bit_1=torch.clone(weight_2bit_1_1)
        #if self.iter==0:
        #    self.var_exp_2_1 = torch.exp(torch.empty(shape_w).normal_(0, 0.2760))
        var_exp_2_1 = torch.exp(torch.empty(shape_w).normal_(0, 0.276))
        weight_2bit_vat1=torch.clone(weight_2bit_1*var_exp_2_1)
        #print_log(torch.linalg.norm(weight_2bit_vat1),log)
        
        #MSB
        weight_2bit_1_2=torch.clone(weight_q)
        weight_2bit_1_2[weight_2bit_1_2.abs()<0.9]=0
        weight_2bit_2=torch.clone(weight_2bit_1_2)
        #if self.iter==0:
        #   self.var_exp_2_2 = torch.exp(torch.empty(shape_w).normal_(0, 0.1035))
        var_exp_2_2 = torch.exp(torch.empty(shape_w).normal_(0, 0.1035))
        weight_2bit_vat2=torch.clone(weight_2bit_2*var_exp_2_2)
        #import pdb;pdb.set_trace()
        weight_q=torch.add(weight_2bit_vat1,weight_2bit_vat2)

        #import pdb;pdb.set_trace()
        #	
    elif w_bit ==3:

        weight_3bit_1_1=torch.clone(weight_q)
        weight_3bit_1_1[weight_3bit_1_1.abs()>0.143]=0
        weight_3bit_1=torch.clone(weight_3bit_1_1)
        #if self.iter==0:
        #	self.var_exp_3_1 = torch.exp(torch.empty(shape_w).normal_(0, 0.3549))
        var_exp_3_1 = torch.exp(torch.empty(shape_w).normal_(0, 0.3549))
        #print(var_exp_3_1.norm())
        weight_3bit_vat1=torch.clone(weight_3bit_1*var_exp_3_1)

        
        #-----------------------2nd bit--------------


        weight_3bit_1_2=torch.clone(weight_q)
        weight_3bit_1_2[weight_3bit_1_2.abs()<0.42]=0
        weight_3bit_1_2[weight_3bit_1_2.abs()>0.43]=0
        
        weight_3bit_2=torch.clone(weight_3bit_1_2)
        #if self.iter==0:

        #	self.var_exp_3_2 = torch.exp(torch.empty(shape_w).normal_(0, 0.2259))
        var_exp_3_2 = torch.exp(torch.empty(shape_w).normal_(0, 0.2259))
        weight_3bit_vat2=torch.clone(weight_3bit_2*var_exp_3_2)

        #-----------------------3rd bit------------------------
        weight_3bit_1_3=torch.clone(weight_q)
        weight_3bit_1_3[weight_3bit_1_3.abs()<0.7]=0
        #import pdb;pdb.set_trace()
        weight_3bit_3=torch.clone(weight_3bit_1_3)

        #if self.iter==0:
        #	self.var_exp_3_3 = torch.exp(torch.empty(shape_w).normal_(0, 0.1898))
        var_exp_3_3 = torch.exp(torch.empty(shape_w).normal_(0, 0.1898))
        weight_3bit_vat3=torch.clone(weight_3bit_3*var_exp_3_3)
        #print(weight_3bit_vat3)

        weight_3bit=torch.add(weight_3bit_vat1,weight_3bit_vat2)
        weight_3bit=torch.add(weight_3bit,weight_3bit_vat3)

        weight_q=weight_3bit
    else:

        weight_4bit_1_1=torch.clone(weight_q)
        weight_4bit_1_1[weight_4bit_1_1.abs()>0.07]=0
        weight_4bit_1=torch.clone(weight_4bit_1_1)
        #if self.iter==0:
        #	self.var_exp_4_1 = torch.exp(torch.empty(shape_w).normal_(0, 0.3549))
        var_exp_4_1 = torch.exp(torch.empty(shape_w).normal_(0, 0.04416))
        #print(var_exp_4_1.norm())
        weight_4bit_vat1=torch.clone(weight_4bit_1*var_exp_4_1)

        
        #-----------------------2nd bit--------------


        weight_4bit_1_2=torch.clone(weight_q)
        weight_4bit_1_2[weight_4bit_1_2.abs()<0.19]=0
        weight_4bit_1_2[weight_4bit_1_2.abs()>0.21]=0
        
        weight_4bit_2=torch.clone(weight_4bit_1_2)

        var_exp_4_2 = torch.exp(torch.empty(shape_w).normal_(0, 0.0513))
        weight_4bit_vat2=torch.clone(weight_4bit_2*var_exp_4_2)

        #-----------------------3rd bit------------------------
        weight_4bit_1_3=torch.clone(weight_q)
        weight_4bit_1_3[weight_4bit_1_3.abs()<0.3]=0
        weight_4bit_1_3[weight_4bit_1_3.abs()>0.47]=0
        #import pdb;pdb.set_trace()
        weight_4bit_3=torch.clone(weight_4bit_1_3)

        var_exp_4_3 = torch.exp(torch.empty(shape_w).normal_(0, 0.06434))
        weight_4bit_vat3=torch.clone(weight_4bit_3*var_exp_4_3)
        #print(weight_4bit_vat3)

        #-----------------------4th bit - part 1- 3 levels------------------------
        weight_4bit_1_4_1=torch.clone(weight_q)
        weight_4bit_1_4_1[weight_4bit_1_4_1.abs()<0.6]=0
        weight_4bit_1_4_1[weight_4bit_1_4_1.abs()>0.7]=0
        #import pdb;pdb.set_trace()
        weight_4bit_4_1=torch.clone(weight_4bit_1_4_1)

        var_exp_4_4_1 = torch.exp(torch.empty(shape_w).normal_(0, 0.069687))
        weight_4bit_vat4_1=torch.clone(weight_4bit_4_1*var_exp_4_4_1)
        #print(weight_4bit_vat3)

        #-----------------------4th bit - part 2- 3 levels------------------------
        weight_4bit_1_4_2=torch.clone(weight_q)
        weight_4bit_1_4_2[weight_4bit_1_4_2.abs()<0.7]=0
        #import pdb;pdb.set_trace()
        weight_4bit_4_2=torch.clone(weight_4bit_1_4_2)

        var_exp_4_4_2 = torch.exp(torch.empty(shape_w).normal_(0, 0.2775))
        weight_4bit_vat4_2=torch.clone(weight_4bit_4_2*var_exp_4_4_2)

        weight_4bit=torch.add(weight_4bit_vat1,weight_4bit_vat2)
        weight_4bit=torch.add(weight_4bit,weight_4bit_vat3)
        weight_4bit=torch.add(weight_4bit,weight_4bit_vat4_1)
        weight_4bit=torch.add(weight_4bit,weight_4bit_vat4_2)
        weight_q=weight_4bit
    
    import pdb;pdb.set_trace()
    
    return weight_q