import os

os.system('python main.py \
			--cell_prec 4\
            --weight_width 8\
            --in_width 8\
            --runs 10\
            --quant\
            --vat\
            --dataset speech_commands\
			--adc_bits 6\
            --xbar_size 1024 \
            --fft_size 4096\
            --pattern 1')