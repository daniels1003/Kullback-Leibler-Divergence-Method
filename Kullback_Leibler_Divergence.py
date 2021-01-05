from __future__ import absolute_import, division, print_function
import numpy as np 
import os
import xlrd
import xlwt 
import openpyxl 
from collections import defaultdict
import operator
import sys
import scipy
from scipy import stats 
import pandas as pd
import itertools


"""
The following function takes in a data frame and a markov or transitional matrix as a 2-D array
and performs the Kullback-Leibler method on the same row of each markov matrix. The Kullback-Leibler
divergence method tells us how similar or different two sets of probability distributions are from each other
Since each row of a markov matrix is a probability distribution of the next state occuring at the next time step
given the current state we can perform the KL method on these distributions. 
"""

def check_probabilities(m,dim): 

	total_probability = 0
	for i, j in itertools.product(range(dim),range(dim)): 
		
		total_probability += m[i][j]
		if m[i][j] < 0:
			raise ValueError("Probability for a state cannot be negative.")
		elif m[i][j] > 1:
			raise ValueError("Probability for a state cannot be greater than 1.")
		elif total_probability != 1:
			raise ValueError("Probability for a row must be equal to 1.")
		elif j == dim: 
			total_probability = 0

def check_equal_matrix_size(markov_matrix,other_markov_matrix): 
	try: 
		if markov_matrix.shape != other_markov_matrix.shape: 
			raise ValueError("matrices are not equal shapes.")
	except: 
		raise 

def check_matrix_dimensions(markov): 
	try: 
		if markov.shape[0] != markov.shape[1]: 
			raise ValueError("Is not markov matrix: matrix dimensions are not equal.")
	except: 
		raise 

def convert_zeros(m,dim): 
	for i, j in itertools.product(range(dim),range(dim)): 
	
			if m[i][j] == 0:
				m[i][j] = 0.00001
			if m[i][j] == 0:
				m[i][j] = 0.00001
			if m[i][j] == 0:
				m[i][j] = 0.00001
	return m

def KL_Divergence_Method(markov_matrix,other_markov_matrix):
	
	check_matrix_dimensions(markov_matrix)
	check_matrix_dimensions(other_markov_matrix)
	check_equal_matrix_size(markov_matrix,other_markov_matrix)
	dim = len(markov_matrix[0])
	check_probabilities(markov_matrix,dim)
	check_probabilities(other_markov_matrix)

	kl = np.zeros(dim)

	markov_matrix = convert_zeros(markov_matrix,dim)
	other_markov_matrix = convert_zeros(other_markov_matrix,dim)
	
	print("Performing KullBack-Leibler Divergence Method...")

	for i in range(dim):
		kl[i] = stats.entropy(pk=markov_matrix[i], qk=other_markov_matrix[i])

	print("KL Method Complete!\n ---------------------------------------")
	print("If the value for the respective row in the matrices is much less than one")
	print("than the difference in the probability distributions is small.")
	print("If the value for the respective row in the matrices is greater than one")
	print("than the difference in the probability distributions is large.")
	print(kl)
