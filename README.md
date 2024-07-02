# SHIFFT_Accuracy_IMC

SHIFFT is a a scalable hybrid in-memory computing FFT accelerator (SHIFFT), a hybrid architecture that combines RRAM-based in-memory computing with CMOS adders. This architecture was developed to overcome the latency limitations seen in traditional CMOS accelerators. The tool is developed to estimate the SQNR and mean absolute error for SHIFFT architecture. The SHIFFT architecture is as follows: the chip architecture comprises an array of IMC tiles, adder trees, and buffers connected using HTree. Each IMC tile is composed of a 2x2 array of processing elements (PE) and an adder tree stage that combines the PE outputs in the same column within the PE array. Each PE, in turn, contains a 2x2 array of IMC crossbar arrays (Xbar) and an adder tree stage that adds the crossbar outputs in the same column within the array of crossbars.

![SHIFFT Architecture](https://github.com/mec-UMN/SHIFFT_Accuracy_IMC/blob/main/SHIFFT%20architecture.jpg)

These codes implement the DFT algorithm, account for inherent quantizations and weight bit slicing in the architecture, and incorporate the statistical training methodology to obtain the maximum quantization value of ADC.

## Usage
To change the configuration of the parameters, edit the file params.py and to run the simulations use the following command:
```
python run_fft.py
```
## References
```
Pragnya Sudershan Nalla, Zhenyu Wang, Sapan Agarwal, T. Patrick Xiao, Christopher H. Bennett, Matthew J. Marinella, Jae-sun Seo, and Yu Cao, SHIFFT: A Scalable Hybrid In-Memory Computing FFT Accelerator, ISVLSI 2024
```

## Developers
Main devs:
* Pragnya Nalla 

Advisors
* Matthew J. Marinella
* Jae-sun Seo
* Yu Cao
