import numpy as np
import math
import matplotlib.pyplot as plt
from param import runs_train, adc_bits, types

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
    save_path="./Figures/SQNR_groove.png"
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print ('---- save figure {} into {}'.format(title, save_path))
    plt.close(fig)

def plot_curve_prec(self):
    title = 'SQNR vs prec'
    dpi = 80  
    width, height = 1200, 800
    legend_fontsize = 10
    scale_distance = 48.8
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    x_axis = 0.1*np.array([i for i in range(types)]) # runs
    y_axis = np.zeros(types)

    plt.xlim(adc_bits[0],adc_bits[types-1])
    plt.ylim(5, 40)
    interval_y = 6
    interval_x = -1
    plt.xticks(np.arange(adc_bits[0],adc_bits[types-1] + interval_x, interval_x))
    plt.yticks(np.arange(5, 40 + interval_y, interval_y))
    plt.grid()
    plt.title(title, fontsize=20)
    plt.xlabel('adc_bits', fontsize=16)
    plt.ylabel('SQNR', fontsize=16)
    
    #import pdb;pdb.set_trace()		
    x_axis[:] = adc_bits
    y_axis[:] = self.prec_SQNR_no_vat_cos[0,:]
    plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-SQNR-prec', lw=2)
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
    save_path="./Figures/SQNR_prec_256_Speech_6bit.png"
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print ('---- save figure {} into {}'.format(title, save_path))
    plt.close(fig)