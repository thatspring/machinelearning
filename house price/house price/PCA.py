# -*- coding:utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import cholesky
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from scipy import optimize as op

def pca(data_array):
	x=np.shape(data_array)
	row=x[0]
	col=x[1]
	#===========================================
	mean_lst=[]
	for i in range(row):
		mean=np.mean(data_array[i,:])
		mean_lst.append(mean)
	mean_data=np.array(mean_lst).reshape(row,1)
	#print "mean:"
	#print mean_data
	#===========================================
	conv_matrix=np.cov(data_array)
	#print "cov_matrix:"
	#print conv_matrix
	#===========================================
	eig_val,eig_vec=np.linalg.eig(conv_matrix)
	#print "eig_val:"
	#print eig_val
	#print "eig_vec:"
	#print eig_vec
	for i in range(len(eig_val)):
		eigv = eig_vec[:,i].reshape(1,row).T
		np.testing.assert_array_almost_equal(conv_matrix.dot(eigv), 
			eig_val[i] * eigv,decimal=6, err_msg='', verbose=True)
	eig=[(eig_val[i],eig_vec[i,:],i) for i in range(len(eig_val))]
	eig.sort(key=lambda x:x[0],reverse=True)
	eigval=-np.sort(-eig_val)
	fig = plt.figure(figsize=(8,8))
	ax = fig.add_subplot(111)
	plt.rcParams['legend.fontsize'] = 10  
	ax.plot(eigval)
	plt.savefig('pca.png')
	#print "eig(sorted):"
	#for i in range(row):
	#	print eig[i][0]
	val0=eigval[0]-eigval[1]
	for i in range(len(eigval)-1):
		val=eigval[i]-eigval[i+1]
		if val/val0<0.01:
			break
	K=i
	print i
	low_matrix=np.hstack((eig[i][1].reshape(row,1) for i in range(K)))
	
	return low_matrix,K,eigval
