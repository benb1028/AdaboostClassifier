# Authors: Ben Brooks and Ryan Marshall
# Created for CS449 Assignment 3 at Clarkson University 
# Adaboost test

import numpy as np
from adaboost import Adaboost
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

def accuracy(y, y_pred):
    acc = np.sum(y == y_pred) / len(y)
    return acc

def error(y, y_pred):
    err = np.sum(y != y_pred) / len(y)
    return err

def make_comparison_plot(max_classifiers, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    tree_model = DecisionTreeClassifier(max_depth = 10)
    svc_model = SVC(C = 1.)
    
    max_classifiers = 25
    n_classifiers = list(range(15, max_classifiers))
    
    tree_acc_list = []
    tree_error_rate_list = []
    
    
    svc_acc_list = []
    svc_error_rate_list = []
    
    skl_default_acc_list = []
    skl_default_error_rate_list = []
    
    for n in n_classifiers:
        
        tree_boost = Adaboost(learner = tree_model, n_classifiers = n)
        tree_boost.fit(X_train, y_train)
        tree_boost.predict(X_test, y_test)
        tree_acc, tree_error_rate = tree_boost.results()
        tree_acc_list.append(tree_acc)
        tree_error_rate_list.append(tree_error_rate)
    
        
        svc_boost = Adaboost(learner = svc_model, n_classifiers = n)
        svc_boost.fit(X_train, y_train)
        svc_boost.predict(X_test, y_test)
        svc_acc, svc_error_rate = svc_boost.results()
        svc_acc_list.append(svc_acc)
        svc_error_rate_list.append(svc_error_rate)
    
    
        skl_default_boost = AdaBoostClassifier(tree_model, n_estimators = n)
        skl_default_boost.fit(X_train, y_train)
        y_pred = skl_default_boost.predict(X_test)    
        skl_default_acc = accuracy(y_test, y_pred)
        skl_default_error_rate = error(y_test, y_pred)
        skl_default_acc_list.append(skl_default_acc)
        skl_default_error_rate_list.append(skl_default_error_rate)
    
        
        print(f'Iteration {n} / {max_classifiers}')
        
    tree_max_acc = max(tree_acc_list)
    tree_min_error = min(tree_error_rate_list)
    svc_max_acc = max(svc_acc_list)
    svc_min_error = min(svc_error_rate_list)
    skl_default_max_acc = max(skl_default_acc_list)
    skl_default_min_error = min(skl_default_error_rate_list)
    print("Best Accuracies by Model:")
    print(f'Tree Model: {tree_max_acc:.4%} with {tree_acc_list.index(max(tree_acc_list))} weak classifiers')
    print(f'SVC Model: {svc_max_acc:.4%} with {svc_acc_list.index(max(svc_acc_list))} weak classifiers')
    print(f'Default Model: {skl_default_max_acc:.4%} with {skl_default_acc_list.index(max(skl_default_acc_list))} weak classifiers')
    fig1 = plt.figure()
    ax = fig1.add_subplot(1,1,1)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    ax.set_xlabel('Number of Classifiers')
    
    ax1 = fig1.add_subplot(2,1,1)
    ax2 = fig1.add_subplot(2,1,2)
    
    ax1.plot(n_classifiers, tree_acc_list, 'g-', n_classifiers, svc_acc_list, 'b-', n_classifiers, skl_default_acc_list, 'r-',)
    ax2.plot(n_classifiers, tree_error_rate_list, 'g-', n_classifiers, svc_error_rate_list, 'b-', n_classifiers, skl_default_error_rate_list, 'r-',)
    
    
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Testing Accuracy of Compared Models')
    ax2.set_ylabel('Test Error Rate')
    ax2.set_title('Test Error Rate of Compared Models')
    #fig1.legend((tree_acc_list, svc_acc_list, skl_default_acc_list),
    #           ('Our Algorithm - Binary Tree Classifier', 'Our Algorithm - SVC',
    #            'Built-in AdaBoost with Binary Tree Classifier'))
    fig1.show()


classes = list(range(0,10))
digits = datasets.load_digits()  
X = digits['data']
y = digits['target']
#y_bin = label_binarize(y, classes)
#print(y[1], y_bin[1])



# boost = Adaboost(learner = tree_model, n_classifiers = 30)
# boost.fit(X_train, y_train)
# boost.predict(X_test, y_test)
# acc, error = boost.results(returnvals = True)
# print(acc, error)
