# Authors: Ben Brooks and Ryan Marshall
# Created for CS449 Assignment 3 at Clarkson University 
# Adaboost test


import numpy as np
from sklearn import datasets
from adaboost import make_result_figure

# import MNIST data 
digits = datasets.load_digits()  
MNIST_X = digits['data']
MNIST_y = digits['target']

# generate figure with MNIST data 
MNIST_fig = make_result_figure('MNIST Handwritten Digits', MNIST_X, MNIST_y)
MNIST_fig.show()

# import banknote data
data = np.loadtxt('data_banknote_authentication.txt', delimiter = ',')
banknote_X = data[:,0:4]
banknote_y = data[:,4]

# generate banknote figure
banknote_fig = make_result_figure('Banknote Authenticity', banknote_X, banknote_y, max_depth = 5, max_itr = 450)
banknote_fig.show()