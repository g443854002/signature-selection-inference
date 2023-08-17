#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abcpy.probabilisticmodels import ProbabilisticModel, Continuous, InputConnector
import numpy as np
from rpy2.robjects.vectors import FloatVector
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

MASS = importr("MASS")
coda = importr("coda")
inline = importr("inline")
Rcpp = importr("Rcpp")
RcppArmadillo = importr("RcppArmadillo")
robjects.r.sourceCpp("./CFUN_1L.cpp")
robjects.r.source("RFUN_1L.R")

'''
This file consist of simulators for 1,2,3 locus wright fisher model.
'''

# os.environ['R_LIBS'] = "/usr/lib/R/library"

class MultiWrightFisher_3loci(ProbabilisticModel, Continuous):
    """
    This class is for 3 loci Wright Fisher Model simulation
    """

    def __init__(self, parameters, population_size=1000, generation=60, interval=10, name='WrightFisher3loci'):
        """
        

        Parameters
        ----------

        population_size : int
            population size. Default value is 1000
        generation : int
            number of generation. Default value is 60
        interval : int
            generation interval, generation should the multiple of the interval .



        Returns
        -------
        Flattened allele frequencies trajetories with selection and dominance.

        """
        if not isinstance(parameters, list):
            raise TypeError('Input of Mimicree model is of type list')

        if len(parameters) != 8:
            raise RuntimeError(
                'Input list must be of length 8, containing [s, h, r].')
           
        self.population_size = population_size
        self.generation = generation
        self.interval = interval
        
        input_connector = InputConnector.from_list(parameters)
        super().__init__(input_connector, name)

    """
    TBA: adapt to fit to the mimicree setup (i.e. file paths) 
    """

    def _check_input(self, input_values):


        return True

    def _check_output(self, values):
        return True

    def get_output_dimension(self):
        return 1

    def forward_simulate(self, input_values, rep, rng=np.random.RandomState()):

        s1 = input_values[0]
        s2 = input_values[1]
        s3 = input_values[2]
        
        h1 = input_values[3]
        h2 = input_values[4]
        h3 = input_values[5]
        
        r1 = input_values[6]
        r2 = input_values[7]
        
        
        s = [float(s1),float(s2),float(s3)]
        h = [float(h1),float(h2),float(h3)]
        r = [float(r1),float(r2)]
        
        #(1/36)*np.linspace(1,8,8)
        #np.array([0.005,0.032,0.196,0.016,0.016,0.132, 0.529,0.074], real starting freq
        #np.array([0.004, 0.02, 0.046, 0.165, 0.026, 0.161, 0.176, 0.401]) mimicree sim freq
        # Do the actual forward simulation
        vector_of_k_samples = self.wfsim(sel_cof=s, dom_par=h, rec=r,
                                         pop_siz=self.population_size,
                                         int_frq=np.array([0.005,0.032,0.196,0.016,0.016,0.132, 0.529,0.074]),
                                         int_gen=0,
                                         lst_gen=self.generation,
                                         interval=self.interval,
                                         num_forward_simulations=rep)
        # Format the output to obey API
        return vector_of_k_samples
    def calculate_allele_freq(self,mat):
        # Define the indices of rows for A1, B1, and C1 frequencies
        a1_rows = [0, 1, 2, 3]
        b1_rows = [0, 1, 4, 5]
        c1_rows = [0, 2, 4, 6]
        
        allele_frequencies = [np.sum(mat[rows, :], axis=0) for rows in [a1_rows, b1_rows, c1_rows]]
        
        return np.vstack(allele_frequencies)
    
    def wfsim(self,sel_cof, dom_par,rec, pop_siz, int_frq, int_gen, lst_gen, interval, num_forward_simulations):
        
        interval_gen = np.linspace(int_gen,lst_gen,int(lst_gen/interval)+1,dtype=int)
        result = []


 
        for i in range(num_forward_simulations):
             result.append(self.compute_haplo_freq(w=self.compute_fitness_matrix_3loci(sel_cof[0],sel_cof[1],sel_cof[2],dom_par[0],dom_par[1],dom_par[2]),
                                              x=int_frq,r_I = rec ,gen=lst_gen,population_size=pop_siz)[:,interval_gen])
 
             

        result = [np.array([self.calculate_allele_freq(x)]).reshape(-1, ) for x in result]
        # put the number of time point as the first value, might be needed for some functions.
        result = [np.insert(np.around(k,3),0,int(lst_gen/interval)+1) for k in result]
        
            
        return result
    def index_to_haplotype(self,index):
        # Given an index (integer), this function converts it into the corresponding
        # haplotype (a tuple of 3 integers representing alleles at each locus).
        # The index ranges from 0 to 7, and the haplotype ranges from (0, 0, 0) to (1, 1, 1).
        return (index // 4, (index % 4) // 2, index % 2)

    def get_haplotype_index(self,h1, h2, I):
        # This function takes two haplotype indices (h1 and h2) and a set I (decomposition of L).
        # It returns the index of the haplotype formed by combining components of h1 and h2
        # based on the decomposition I.

        # Convert the indices h1 and h2 to their corresponding haplotypes.
        h1_haplotype = self.index_to_haplotype(h1)
        h2_haplotype = self.index_to_haplotype(h2)

        # Create the resulting haplotype by choosing components from h1_haplotype if the
        # index is in I, and from h2_haplotype if the index is not in I.
        result = [h1_haplotype[i] if i in I else h2_haplotype[i] for i in range(3)]

        # Convert the resulting haplotype back to an index, which is returned.
        return result[0] * 4 + result[1] * 2 + result[2]
    
    def compute_fitness_matrix_3loci(self,s_a, s_b, s_c, h_a, h_b, h_c):
        # compute the fitness matrix when we have 3 loci
        w_A1A1 = 1
        w_A1A2 = 1 - h_a * s_a
        w_A2A2 = 1 - s_a
        w_B1B1 = 1
        w_B1B2 = 1 - h_b * s_b
        w_B2B2 = 1 - s_b
        w_C1C1 = 1
        w_C1C2 = 1 - h_c * s_c
        w_C2C2 = 1 - s_c
        

    
        w_A = np.array([[w_A1A1, w_A1A2], [w_A1A2, w_A2A2]])
        w_B = np.array([[w_B1B1, w_B1B2], [w_B1B2, w_B2B2]])
        w_C = np.array([[w_C1C1, w_C1C2], [w_C1C2, w_C2C2]])
        
        w_A = w_A.reshape(2, 1, 1, 2, 1, 1)
        w_B = w_B.reshape(1, 2, 1, 1, 2, 1)
        w_C = w_C.reshape(1, 1, 2, 1, 1, 2)
    
        w = (w_A * w_B * w_C).reshape(8, 8)
        return w
    
    def compute_recombination_term(self, w, x, r):
        """
        Computes the recombination term.
    
        Args:
            w (numpy.ndarray): The fitness matrix for the haplotypes.
            x (numpy.ndarray): An array containing the haplotype frequencies.
            r (dict): A dictionary of recombination rates for each decomposition.
    
        Returns:
            numpy.ndarray: A 1D array representing the recombination term for each haplotype.
        """
        # Initialize an array of zeros to store the recombination terms for each haplotype.
        thetas = np.zeros(8)
        
        r_AB = r[0]
        r_BC = r[1]
        # Iterate over all haplotypes.
        for i in range(8):
            theta_i = 0
            # Iterate over all possible pairs of haplotypes.
            for j in range(8):
                # Iterate over all possible decompositions of L.
                for I, r_I in zip([frozenset({0}), frozenset({1}), frozenset({2}), frozenset({0, 1}), frozenset({0, 2}), frozenset({1, 2})],
                  [r_AB, r_AB * r_BC, r_BC, r_BC, r_AB * r_BC, r_AB]):
                    # Compute the complement of the decomposition I, which is J.
                    J = frozenset({0, 1, 2}) - I
    
                    # Get the indices of the haplotypes formed by combining components
                    # of haplotypes i and j based on the decomposition I.
                    i_Ij_J_index = self.get_haplotype_index(i, j, I)
                    j_Ii_J_index = self.get_haplotype_index(j, i, I)
    
                    # Compute the contribution of the current haplotype pair i and j
                    # to the recombination term for haplotype i.
                    #print(x[i_Ij_J_index])
                    theta_i += w[i, j] * r_I * (x[i] * x[j] - x[i_Ij_J_index] * x[j_Ii_J_index])
    
            # Normalize the recombination term for haplotype i by dividing by the sum of
            # the outer product of the haplotype frequencies multiplied by the fitness matrix.
            thetas[i] = theta_i / np.sum(w * np.outer(x, x))
    
        # Return the recombination terms for all haplotypes.
        return thetas
    
    def compute_haplo_freq(self, w, x, r_I, gen=60, population_size=1000):
        
        """
        Calculates the haplotype frequencies over time using a fitness matrix and starting haplotype frequencies.

        Args:
            w (numpy.ndarray): The fitness matrix for the haplotypes.
            x (numpy.ndarray): An array containing the starting frequencies of the haplotypes.
            r_I (float): Recombination rate.
            gen (int): The number of generations to simulate.

        Returns:
            numpy.ndarray: A 2D array with dimensions (gen, 8) representing the haplotype frequencies at each generation.
        """
        freq = np.zeros((gen + 1, 8))
        freq[0, :] = x

        for i in range(gen):
            thetas = self.compute_recombination_term(w, x, r_I)
            numerator = np.sum(w * x, axis=1)
           
            denominator = np.sum(w * np.outer(x, x))
            
            x_prime = (numerator / denominator) * x - thetas
            x_prime = x_prime/np.sum(x_prime)
            x = np.random.multinomial(2 * population_size, x_prime) / (2 * population_size)
            freq[i + 1, :] = x

        return freq.T




    
class WrightFisher_c(ProbabilisticModel, Continuous):
    """
    This class using He's method to simulate in 1-locus case with Wright-Fisher diffusion.
    """

    def __init__(self, parameters, population_size=1000, generation=60, interval=10, name='WrightFisherpy'):
        """
        

        Parameters
        ----------
        rmodel: model
            specify the model wrapper for simulation
        population_size : int
            population size. Default value is 1000
        generation : int
            number of generation. Default value is 60
        interval : int
            generation interval, generation should the multiple of the interval .



        Returns
        -------
        Flattened allele frequencies trajetories with selection and dominance.

        """
        if not isinstance(parameters, list):
            raise TypeError('Input of Mimicree model is of type list')

        if len(parameters) != 2:
            raise RuntimeError(
                'Input list must be of length 2, containing [s, h].')
           
        self.population_size = population_size
        self.generation = generation
        self.interval = interval

        
        input_connector = InputConnector.from_list(parameters)
        super().__init__(input_connector, name)

    """
    TBA: adapt to fit to the mimicree setup (i.e. file paths) 
    """

    def _check_input(self, input_values):


        return True

    def _check_output(self, values):
        return True

    def get_output_dimension(self):
        return 1

    def forward_simulate(self, input_values, rep, rng=np.random.RandomState()):

        s = input_values[0]
        h = input_values[1]

        # Do the actual forward simulation
        vector_of_k_samples = self.wfsim(sel_cof=s, dom_par=h,
                                         pop_siz=self.population_size,
                                         int_frq=0.2,int_gen=0,
                                         lst_gen=self.generation,
                                         interval=self.interval,num_forward_simulations=rep)
        # Format the output to obey API
        return vector_of_k_samples

    def wfsim(self,sel_cof, dom_par, pop_siz, int_frq, int_gen, lst_gen, interval, num_forward_simulations):
        cmpsimulateOLWFDS = robjects.r['cmpsimulateOLWFDS']
        interval_gen = np.linspace(int_gen,lst_gen,int(lst_gen/interval)+1,dtype=int)
        
        result = []
        
        sel_cof = float(sel_cof)
        dom_par = float(dom_par)
        
        
        for i in range(num_forward_simulations):
            
             result.append(np.array(cmpsimulateOLWFDS(sel_cof, dom_par, pop_siz, int_frq, int_gen, lst_gen, ptn_num=1, dat_aug = False))[interval_gen])
        
        
        result = [np.array([x]).reshape(-1, ) for x in result]
        # put the number of time point as the first value, might be needed for some functions.
        result = [np.insert(np.around(k,3),0,int(lst_gen/interval)+1) for k in result]
        
            
        return result 
    
class WrightFisher_b(ProbabilisticModel, Continuous):
    """
    This class using He's method to simulate in 1-locus case with standard Wright-Fisher.

    """

    def __init__(self, parameters, population_size=1000, generation=60, interval=10, name='WrightFisherpy'):
        """
        

        Parameters
        ----------
        rmodel: model
            specify the model wrapper for simulation
        population_size : int
            population size. Default value is 1000
        generation : int
            number of generation. Default value is 60
        interval : int
            generation interval, generation should the multiple of the interval .



        Returns
        -------
        Flattened allele frequencies trajetories with selection and dominance.

        """
        if not isinstance(parameters, list):
            raise TypeError('Input of Mimicree model is of type list')

        if len(parameters) != 2:
            raise RuntimeError(
                'Input list must be of length 2, containing [s, h].')
           
        self.population_size = population_size
        self.generation = generation
        self.interval = interval
        
        input_connector = InputConnector.from_list(parameters)
        super().__init__(input_connector, name)

    """
    TBA: adapt to fit to the mimicree setup (i.e. file paths) 
    """

    def _check_input(self, input_values):


        return True

    def _check_output(self, values):
        return True

    def get_output_dimension(self):
        return 1

    def forward_simulate(self, input_values, rep, rng=np.random.RandomState()):

        s = input_values[0]
        h = input_values[1]

        # Do the actual forward simulation
        vector_of_k_samples = self.wfsim(sel_cof=s, dom_par=h,
                                         pop_siz=self.population_size,
                                         int_frq=0.2,int_gen=0,
                                         lst_gen=self.generation,
                                         interval=self.interval,num_forward_simulations=rep)
        # Format the output to obey API
        return vector_of_k_samples

    def wfsim(self, sel_cof, dom_par, pop_siz, int_frq, int_gen, lst_gen, interval, num_forward_simulations):
        cmpsimulateOLWFMS = robjects.r['cmpsimulateOLWFMS']
        interval_gen = np.linspace(int_gen,lst_gen,int(lst_gen/interval)+1,dtype=int)
        
        result = []
        
        sel_cof = float(sel_cof)
        dom_par = float(dom_par)
        
        
        for i in range(num_forward_simulations):

             result.append(np.array(cmpsimulateOLWFMS(sel_cof, dom_par, pop_siz, int_frq, int_gen, lst_gen))[interval_gen])

        
        
        result = [np.array([x]).reshape(-1, ) for x in result]
        # put the number of time point as the first value, might be needed for some functions.
        result = [np.insert(np.around(k,3),0,int(lst_gen/interval)+1) for k in result]
        
            
        return result     

class MultiWrightFisher_c(ProbabilisticModel, Continuous):
    """
    This class using He's method do 2-locus simulation with Wright-Fisher diffusion.
    """

    def __init__(self, parameters,rmodel, population_size=1000, generation=60, interval=10, name='WrightFisherpy'):
        """
        

        Parameters
        ----------
        rmodel: model
            specify the model wrapper for simulation
        population_size : int
            population size. Default value is 1000
        generation : int
            number of generation. Default value is 60
        interval : int
            generation interval, generation should the multiple of the interval .



        Returns
        -------
        Flattened allele frequencies trajetories with selection and dominance.

        """
        if not isinstance(parameters, list):
            raise TypeError('Input of Mimicree model is of type list')

        if len(parameters) != 4:
            raise RuntimeError(
                'Input list must be of length 4, containing [s, h].')
           
        self.population_size = population_size
        self.generation = generation
        self.interval = interval
        self.r_model = rmodel
        
        input_connector = InputConnector.from_list(parameters)
        super().__init__(input_connector, name)

    """
    TBA: adapt to fit to the mimicree setup (i.e. file paths) 
    """

    def _check_input(self, input_values):


        return True

    def _check_output(self, values):
        return True

    def get_output_dimension(self):
        return 1

    def forward_simulate(self, input_values, rep, rng=np.random.RandomState()):

        s1 = input_values[0]
        s2 = input_values[1]
        
        h1 = input_values[2]
        h2 = input_values[3]
        
        s = FloatVector([float(s1),float(s2)])
        h = FloatVector([float(h1),float(h2)])
        

        # Do the actual forward simulation
        vector_of_k_samples = self.wfsim(r_wf=self.r_model,sel_cof=s, dom_par=h,
                                         pop_siz=self.population_size,
                                         int_frq=FloatVector(np.asarray([0.1,0.2,0.3,0.4])),int_gen=0,
                                         lst_gen=self.generation,
                                         interval=self.interval,num_forward_simulations=rep,
                                         snp_num=self.snp_size)
        # Format the output to obey API
        return vector_of_k_samples

    def wfsim(self, r_wf,sel_cof, dom_par, pop_siz, int_frq, int_gen, lst_gen, interval, num_forward_simulations,snp_num):
        
        interval_gen = np.linspace(int_gen,lst_gen,int(lst_gen/interval)+1,dtype=int)
        result = []

        # recombination rate
        rec_rat = 0.00001
        
        for i in range(num_forward_simulations):
            
             result.append(np.array(r_wf(sel_cof, dom_par, rec_rat, pop_siz, int_frq, int_gen, lst_gen, ptn_num=1, dat_aug = False))[:,interval_gen])

        
        
        
        result = [np.array([x]).reshape(-1, ) for x in result]
        # put the number of time point as the first value, might be needed for some functions.
        result = [np.insert(np.around(k,3),0,int(lst_gen/interval)+1) for k in result]
        
            
        return result    
    
class MultiWrightFisher_b(ProbabilisticModel, Continuous):
    """
    This class using He's method do 2-locus simulation with Wright-Fisher diffusion.

    """

    def __init__(self, parameters,rmodel, population_size=1000, generation=60, interval=10, name='WrightFisherpy'):
        """
        

        Parameters
        ----------
        rmodel: model
            specify the model wrapper for simulation
        population_size : int
            population size. Default value is 1000
        generation : int
            number of generation. Default value is 60
        interval : int
            generation interval, generation should the multiple of the interval .
        snp_size : int
            number of snps.


        Returns
        -------
        Flattened allele frequencies trajetories with selection and dominance.

        """
        if not isinstance(parameters, list):
            raise TypeError('Input of Mimicree model is of type list')

        if len(parameters) != 4:
            raise RuntimeError(
                'Input list must be of length 4, containing [s, h].')
           
        self.population_size = population_size
        self.generation = generation
        self.interval = interval
        self.r_model = rmodel
        
        input_connector = InputConnector.from_list(parameters)
        super().__init__(input_connector, name)

    """
    TBA: adapt to fit to the mimicree setup (i.e. file paths) 
    """

    def _check_input(self, input_values):


        return True

    def _check_output(self, values):
        return True

    def get_output_dimension(self):
        return 1

    def forward_simulate(self, input_values, rep, rng=np.random.RandomState()):

        s1 = input_values[0]
        s2 = input_values[1]
        
        h1 = input_values[2]
        h2 = input_values[3]
        
        s = FloatVector([float(s1),float(s2)])
        h = FloatVector([float(h1),float(h2)])
        

        # Do the actual forward simulation
        vector_of_k_samples = self.wfsim(r_wf=self.r_model,sel_cof=s, dom_par=h,
                                         pop_siz=self.population_size,
                                         int_frq=FloatVector(np.asarray([0.166 ,0.222 ,0.278 ,0.334])),int_gen=0,
                                         lst_gen=self.generation,
                                         interval=self.interval,num_forward_simulations=rep)
        # Format the output to obey API
        return vector_of_k_samples
    def calculate_allele_freq(self,mat):
        # Define the indices of rows for A1, B1, and C1 frequencies
        a1_rows = [0, 1]
        b1_rows = [0, 2]
        
        allele_frequencies = [np.sum(mat[rows, :], axis=0) for rows in [a1_rows, b1_rows]]
        
        return np.vstack(allele_frequencies)
    def wfsim(self, r_wf,sel_cof, dom_par, pop_siz, int_frq, int_gen, lst_gen, interval, num_forward_simulations,snp_num):
        
        interval_gen = np.linspace(int_gen,lst_gen,int(lst_gen/interval)+1,dtype=int)
        result = []
        
        # real data generations
        # interval_gen = np.array([0 ,16,20,26,30,71,76,83,89 ,95, 108 ,119 ,126, 152, 156, 157, 164, 176, 183 , 
        #                          187,203,214,220 ,226, 232, 289, 326, 333 ,352, 358 ,362 ,370,
        #                          376 ,380, 383, 391, 398, 402,408])
        rec_rat = 0.00008869
        
        for i in range(num_forward_simulations):
             result.append(np.array(r_wf(sel_cof, dom_par, rec_rat, pop_siz, int_frq, int_gen, lst_gen))[:,interval_gen])
        
        
        
        result = [np.array([self.calculate_allele_freq(x)]).reshape(-1, ) for x in result]
        # put the number of time point as the first value, might be needed for some functions.
        result = [np.insert(np.around(k,3),0,int(lst_gen/interval)+1) for k in result]
        
            
        return result  

    