# Authors: Ben Brooks and Ryan Marshall
# Created for CS449 Assignment 3 at Clarkson University 
# Adaboost test


# max_depth for decision tree estimator needs to be at least 4. After testing,
# max_depth = 10 provided near maximum testing accuracy without overfitting.

# learn_rate can be from 0-1, but high values are not effective. After testing,
# learn_rate = 0.05 was chosen, but rates from 0.1 to 0.0001 provided similar test accuracy.

import numpy as np


from sklearn import datasets


from adaboost import make_result_figure
'''
digits = datasets.load_digits()  
MNIST_X = digits['data']
MNIST_y = digits['target']

MNIST_fig = make_result_figure('MNIST Handwritten Digits', MNIST_X, MNIST_y)
MNIST_fig.show()
'''

data = np.loadtxt('data_banknote_authentication.txt', delimiter = ',')
banknote_X = data[:,0:4]
banknote_y = data[:,4]
#print(banknote_X.shape, len(banknote_y))
print(banknote_y)
banknote_fig = make_result_figure('Banknote Authenticity', banknote_X, banknote_y, max_depth = 5, max_itr = 1500)
banknote_fig.show()