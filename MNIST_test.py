# Authors: Ben Brooks and Ryan Marshall
# Created for CS449 Assignment 3 at Clarkson University 
# Adaboost test


# max_depth for decision tree estimator needs to be at least 4. After testing,
# max_depth = 10 provided near maximum testing accuracy without overfitting.

# learn_rate can be from 0-1, but high values are not effective. After testing,
# learn_rate = 0.05 was chosen, but rates from 0.1 to 0.0001 provided similar test accuracy.

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from adaboost import make_result_figure

digits = datasets.load_digits()  
MNIST_X = digits['data']
MNIST_y = digits['target']

MNIST_fig = make_result_figure('MNIST Handwritten Digits', MNIST_X, MNIST_y)
MNIST_fig.show()

'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

max_depth = 10
learn_rate = 0.05
n_classifiers = 80
max_itr = 1800

#accs = compare_learn_rates(X = X, y = y, learn_rates = learn_rates, num_cycles = num_cycles, max_depth = max_depth)
tree_base = DecisionTreeClassifier(max_depth = max_depth)
logreg_base = LogisticRegression(max_iter = max_itr)

custom_tree_model = Adaboost(learner = tree_base, n_classifiers = n_classifiers, learn_rate = learn_rate).fit(X_train, y_train)
custom_tree_model.predict(X_test, y_test)
ctm_acc, ctm_train_error = custom_tree_model.results()
ctm_train_error = fix_train_error(ctm_train_error)

custom_logreg_model = Adaboost(learner = logreg_base, n_classifiers = n_classifiers, learn_rate = learn_rate).fit(X_train, y_train)
custom_logreg_model.predict(X_test, y_test)
clrm_acc, clrm_train_error = custom_logreg_model.results()
clrm_train_error = fix_train_error(clrm_train_error)

skl_tree_model = AdaBoostClassifier(tree_base, n_estimators = n_classifiers, learning_rate = learn_rate).fit(X_train, y_train)
skltm_acc = skl_tree_model.score(X_test, y_test)
skltm_train_error = fix_train_error(skl_tree_model.estimator_errors_)

skl_logreg_model = AdaBoostClassifier(logreg_base, n_estimators = n_classifiers, learning_rate = learn_rate, algorithm = 'SAMME').fit(X_train, y_train)
skllrm_acc = skl_logreg_model.score(X_test, y_test)
skllrm_train_error = fix_train_error(skl_logreg_model.estimator_errors_)

models = {'0':"Custom Implementation with binary tree classifier",
          '1':"Custom Implementation with logistic regression classifier",
          '2':"SKLearn Implementation with binary tree classifier",
          '3':"SKLearn Implementation with logistic regression classifier"}
labels = [f'Model {i}: {label}' for (i, label) in models.items()]
legend_str = '\n'.join([label for label in labels])
accuracies = [ctm_acc, clrm_acc, skltm_acc, skllrm_acc]
avg_acc = np.mean(accuracies)
classifiers = np.arange(1, n_classifiers + 1)
param_string = f"* All models trained with learning rate of {learn_rate}. Tree classifiers have max depth of {max_depth}. Regression classifiers have {max_itr} maximum iterations."

grid_size = (4, 6)
fig = plt.figure(figsize=(18, 12))
ax1 = plt.subplot2grid(grid_size, (0,0), colspan = 3, rowspan = 3)
ax2 = plt.subplot2grid(grid_size, (0,3), colspan = 3, rowspan = 3)
ax3 = plt.subplot2grid(grid_size, (3,0), colspan = 6, rowspan = 1)

ax1.plot(classifiers, ctm_train_error, marker = 'o', markersize = 16, color = 'r', linewidth = 0.75, label = 'Model 0', alpha = 0.75)
ax1.plot(classifiers, clrm_train_error, marker = '+', markersize = 16, color = 'r', linewidth = 0.75, label = 'Model 1')
ax1.plot(classifiers, skltm_train_error, marker = 'o', markersize = 16, color = 'b', linewidth = 0.75, label = 'Model 2', alpha = 0.75)
ax1.plot(classifiers, skllrm_train_error, marker = 'x', markersize = 16, color = 'b', linewidth = 0.75, label = 'Model 3')

ax1.set_title('Training Error vs. # of Classifiers', fontsize = 18)
ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax1.set_xlabel('Classifiers', fontsize = 16)
ax1.set_ylabel('Error', fontsize = 16)
ax1.legend(loc = 'upper right', fontsize = 16)

ax2.bar(list(models.keys()), accuracies, color = ['r', 'r', 'b', 'b'])
ax2.plot([-0.5, 3.5], [avg_acc]*2, marker = '', color = 'k', linestyle = 'dashed', linewidth = 4.0)
ax2.text(-0.62, avg_acc + 0.02, f'Average = {avg_acc:.2%}', fontsize = 16)

ax2.set_title('Testing Accuracy of Models', fontsize = 18)
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax2.set_xlabel('Model', fontsize = 16)
ax2.set_ylabel('Accuracy', fontsize = 16)

ax3.set_title('Models:', fontsize = 18)
ax3.axis([0, 1, 0, 1])
ax3.get_xaxis().set_visible(False)
ax3.get_yaxis().set_visible(False)

ax3.text(0.05, 0.80, labels[0], transform=ax3.transAxes, fontsize = 14)
ax3.text(0.05, 0.50, labels[1], transform=ax3.transAxes, fontsize = 14)
ax3.text(0.55, 0.80, labels[2], transform=ax3.transAxes, fontsize = 14)
ax3.text(0.55, 0.50, labels[3], transform=ax3.transAxes, fontsize = 14)
ax3.text(0.50, 0.10, param_string, transform=ax3.transAxes, fontsize = 14, horizontalalignment = 'center')

fig.suptitle('MNIST Handwritten Digit Test: Results', x = 0.5, y = 1.03, fontsize = 24)
fig.tight_layout(pad = 0.5, w_pad = 0.5, h_pad = 0.5)
fig.show()
'''

