# Implementation of Kernel PCA and PCA.
# Author : Ashutosh Kakadiya

from __future__ import division

from scipy import exp
from scipy.linalg import eigh
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def read_data(filename):
	data=pd.read_csv(filename,header=None)
	x=np.array(data)
	return x

# Calculate Variance
def cal_variance(a):
    '''
    Input : a is dxd numpy matrix
    Output : A variance vector of dimention d
    '''
    n,d = np.shape(a)
    mean_matrix = np.ones((n,d)) * cal_mean(a)
    variance =(1/n)* np.diag(np.transpose(a-mean_matrix).dot(a-mean_matrix))
    return variance


#Calculate Covariance Matrix
def cal_covariance(a):
    '''
    Input : a is nxd numpy matrix
    Output : A covariance matrix of dimention dxd
    '''
    n,d = np.shape(a)
    mean_matrix = np.ones((n,d))*cal_mean(a)
    co_var = (1/(n-1)) * (np.transpose(a-mean_matrix).dot(a-mean_matrix))
    return co_var


#Calculate Mean
def cal_mean(a):
    '''
    Input : a is nxd numpy matrix
    Output : A mean vector of dimention d
    '''
    n,d = np.shape(a)
    #print(n,d)
    mean = np.sum(a,axis=0) / n
    return(mean)

# Calculate Retain Variance by eigen values.
def cal_retain_var(evalues,n_comp):
	total_sum = np.sum(evalues)
	var_ret = (evalues/total_sum)*100
	var_retain = np.cumsum(var_ret)  # Cumulative sum.
	return var_retain[n_comp-1]  # return retain variance of top n components.

#Calculate Kernel A given in Assignment
def kernel_A(x,y,d):
    temp = x.T.dot(y)
    return((1+temp)**d)

# Main Kernel function A
def main_kernel_A(x,d):
	temp=[]
	kernel = np.zeros([x.shape[0],x.shape[0]])

	for i in range(x.shape[0]-1):
		for j in range(i,x.shape[0]):
			temp=kernel_B(x[i],x[j],d)
			kernel[i][j]=temp
			kernel[j][i]=temp

	return kernel

#Calculate Kernel B given in Assignment (rbf kernel)
def kernel_B(x,y,sigma):
    temp = (-1 * ((x-y).T.dot(x-y))) / (2*sigma**2)
    return(np.exp(temp))

# Main Kernel function A
def main_kernel_B(x,sigma):
	temp=[]
	kernel = np.zeros([x.shape[0],x.shape[0]])

	for i in range(x.shape[0]-1):
		for j in range(i,x.shape[0]):
			temp=kernel_B(x[i],x[j],sigma)
			kernel[i][j]=temp
			kernel[j][i]=temp

	return kernel


# Centering the Kernel Matrix
def center_kernel(kernel):
	
	n = kernel.shape[0]
	sum_rows = np.sum(kernel, axis=0) / n 	
	sum_all = sum_rows.sum() / n
	sum_column = (np.sum(kernel,axis=1) / sum_rows.shape[0])[:,np.newaxis] 

	kernel-= sum_rows
	kernel-= sum_column
	kernel+= sum_all

	return kernel


# Main function of KPCA
def Cal_kernel(x,kernel_type):
	kernel =[]
	# Computation of kernel matrix 
	if kernel_type=="A":
		d = 2   # pass value of d here
		kernel = main_kernel_A(x,d) 

	elif kernel_type=="B":
		std=3
		kernel = main_kernel_B(x,std)

	# Centering the Kernel Matrix
	ckernel = center_kernel(kernel)

	return ckernel
	# Return Centered Kernel Matrix.

def kpca(x,kernel_type,n_comp):

	kernel = Cal_kernel(x,kernel_type)
	
	# Calculating Eigen Value and Eigen Vector (function returns in non-increasing order)
	evalues,evectors = eigh(kernel)

	# Sorting eigen values in increasing order.
	idex = evalues.argsort()[::-1]
	evalues = evalues[idex]

	top_evalues = evalues[:n_comp]   # top n eigen values.
	top_evectors = np.atleast_1d(evectors[:,idex])[:,:n_comp]

	# Project the data into reduced subspace.
	new_x = top_evectors * np.sqrt(top_evalues)  # A*x = lambda * x So, we dont haveto use kernel matrix.
	#new_x = kernel.dot(evectors)

	# Calculate retain variance # evalues is already sorted in decreasing order.
	retain_var = cal_retain_var(evalues,n_comp) 
	#print(retain_var)


	return new_x,retain_var


# Main function of PCA
def pca(x,n_comp):
	assert n_comp <= x.shape[1], "total n components must be less or equal to dimention of data"
	# Calculating Covariance Matrix.
	covar_m = cal_covariance(x)

	# Calculating Eigen Value and Eigen Vector (function returns in non-increasing order)
	evalues,evectors = np.linalg.eig(covar_m)

	# Sorting eigen values in increasing order.
	idex = evalues.argsort()[::-1]
	evalues = evalues[idex]
	top_evalues = evalues[:n_comp]   # top n eigen values.
	evectors = np.atleast_1d(evectors[:,idex])[:,:n_comp]

	# Project the data into reduced subspace.
	new_x = x.dot(evectors)

	# Calculate retain variance # evalues is already sorted in decreasing order.
	retain_var = cal_retain_var(evalues,n_comp) 
	#print(retain_var)

	return new_x,retain_var


def plot_data(data,std):
	plt.style.use('seaborn-whitegrid')
	ax = sns.scatterplot(x=data[:,0],y=data[:,1],color='orange')
	plt.title("Scatter plot : RBF Kernel, sigma = "+str(std))
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.savefig("rbf_kernel_"+str(std)+"_sigma= .png")
	plt.close()

if __name__ == "__main__":
	file_path = "Data/Dataset3.csv"
	x=read_data(file_path) # Read the data. Return numpy array

	n_comp = 2   # number to retain top n components
	#Project_x,retain_var = pca(x,n_comp)
	

	kernel_type = "B"  # A for Gaussian Kernel, B for linear multiplication
	Project_x,retain_var = kpca(x,kernel_type,n_comp)	
	#plot_data(Project_x)
	#plot_data(x)

