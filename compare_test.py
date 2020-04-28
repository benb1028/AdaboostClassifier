# Author: Ben Brooks
# Date of Creation:
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from model_comparison import Model
from model_comparison import compare_models


model1 = Model('Decision Tree - Depth 1', DecisionTreeClassifier(max_depth = 1))
model2 = Model('Decision Tree - Depth 5', DecisionTreeClassifier(max_depth = 5))
model3 = Model('Decision Tree - Depth 10', DecisionTreeClassifier(max_depth = 10))
model4 = Model('Logistic Regression', LogisticRegression(max_iter = 1800))

dataset_list = []
models = [model1, model2, model3, model4]

digits = datasets.load_digits()
dataset_list.append((digits['data'], digits['target']))

banknote_data = np.loadtxt('data_banknote_authentication.txt', delimiter = ',')
dataset_list.append((banknote_data[:,0:4], banknote_data[:,4]))


results = compare_models(dataset_list, models)
print(results.head(20))
print(results.tail(20))
