#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from approx_lik import SigScore
from abcpy.statistics import Identity
from abcpy.inferences import PMC
from abcpy.backends import BackendMPI as Backend
from abcpy.perturbationkernel import DefaultKernel
from abcpy.continuousmodels import Uniform
from WrightFisher_simulator import WrightFisher_b,MultiWrightFisher_c,WrightFisher_c,MultiWrightFisher_3loci,MultiWrightFisher_b
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import rpy2.robjects as robjects

'''
This file gives an example how to run PMC sampler with different simulators.s
'''

# Load simulator from R
# See He, Zhangyi, et al. "Detecting and quantifying natural selection at two 
# linked loci from time series data of allele frequencies with forward-in-time simulations."
# Genetics 216.2 (2020): 521-541 for more details


try:
    robjects.r('''
        source('RFUN_2L.R')
    ''')
except:
    robjects.r('''
            source('RFUN_2L.R')
    ''')
r_wf = robjects.globalenv['cmpsimulateTLWFMS']

# Specify prior for s

s_prior1 = Uniform([[0], [0.2]], name='s_prior1')
s_prior2 = Uniform([[0], [0.2]], name='s_prior2')
s_prior3 = Uniform([[0], [0.2]], name='s_prior3')



# 2-locus with standard wright fisher model
WFS = MultiWrightFisher_b([s_prior1, s_prior2,0.5,0.5],rmodel=r_wf,population_size=996, 
                      generation=100, interval=10, name='WFS')



# 3-locus wright fisher simulator
WFS = MultiWrightFisher_3loci([s_prior1, s_prior2,s_prior3,0.5,0.5,0.5,0.1,0.001],population_size=300, 
                       generation=100, interval=10, name='WFS3')
# specify backend for parallel computing
backend = Backend()
# specify PMC parameters
n_sample = 300
n_samples_per_param = 50
step = 5

#load real data
data = [np.load('data.npz')['d']]
#specify likelihood computation
lik_hd = SigScore(Identity(degree=1, cross=False))
#define kernel for parameter of interest
kernel = DefaultKernel([s_prior1,s_prior2,s_prior3])
#define sampler with the tuning parameter
sampler = PMC([WFS],[lik_hd], backend, kernel, seed=1)
#sample from the sampler
journal = sampler.sample([data],steps=step, n_samples=n_sample, 
                                      n_samples_per_param=n_samples_per_param,
                                      full_output=1)
#save the result
file = 'file.jrnl'
journal.save(file)

        
        
        
