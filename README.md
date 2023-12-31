# Signature-Informed Selection Detection: A Novel Method for Multi-Locus Wright-Fisher Models with Recombination

This repository houses Python code developed to simulate allele frequencies following the Wright-Fisher Model. 
In addition, it includes a PMC (Population Monte Carlo) sampler. And the code for calculating the signature and use it as the likelihood approximation are included. This code is for the paper "Signature-Informed Selection Detection: A Novel Method for Multi-Locus Wright-Fisher Models with Recombination"

## File Descriptions:

1. **approx_lik.py**: Implements three scoring rule calculations:
   - Energy Score
   - Kernel Score
   - Signature Kernel Scoring Rule
   
   For more details on the Signature Kernel, refer to the following paper:
   
@article{salvi2020computing,
title={The Signature Kernel is the solution of a Goursat PDE},
author={Salvi, Cristopher and Cass, Thomas and Foster, James and Lyons, Terry and Yang, Weixin},
journal={arXiv preprint arXiv:2006.14794},
year={2020}
}

2. **Sampler.py**: Demonstrates how to execute sampling for our paper using the simulator and the PMC sampler from the ABCpy package.

3. **Wrightfisher_simulator.py**: Houses simulators for:
- 1-locus Wright-Fisher
- 2-locus Wright-Fisher
- 3-locus Wright-Fisher

Note that the R files and functions for simulations are from https://github.com/zhangyi-he/WFM-2L-DiffusApprox-PMMH by the paper titled "Detecting and Quantifying Natural Selection at Two Linked Loci from Time Series Data of Allele Frequencies with Forward-in-Time Simulations"

