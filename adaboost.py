# Authors: Ben Brooks and Ryan Marshall
# Created for CS449 Assignment 3 at Clarkson University 
# Adaboost Implementation

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


class CLF:
    '''Generic classifier to copy'''
    
    def __init__(self, learner):
        self.classifier = learner
        
        
class Adaboost:
    '''Define the class used for boosting.
    
    Contains any methods neccessary for the algorithm and stores the 
    relevent data for analysis and prediction.
    '''
    
    def __init__(self, learner, n_classifiers = 20, learn_rate = 0.1):
        '''Initialize the class with optional parameters.'''
        self.n_classifiers = n_classifiers
        self.classifiers = []
        self.learner = learner
        self.train_accuracy = []
        self.accuracy = None
        self.predictions = []
        self.bin_preds = []
        self.y_test = []
        self.error_rate = None
        if learn_rate > 1 or learn_rate <=0:
            print("Warning: invalid learning rate. Learn rate has been set to 1.")
            self.learn_rate = 1
        else:
            self.learn_rate = learn_rate
        
    def fit(self, X, y):
        '''Fit the model to a training sample.
        
        Stores the array of weak classifiers used for prediction,
        as well as their weights and the sample weights.
        '''
        sample_n, n_variables = X.shape
        w = np.full(sample_n, (1./sample_n))
        self.alphas = np.zeros(self.n_classifiers, dtype = np.float64)
        self.classifier_errors = np.ones(self.n_classifiers, dtype = np.float64)
        
        for idx in range(self.n_classifiers):
            w, alpha, classifier_error = self.boost(idx, X, y, w)
            
            if w is None:
                break
            
            self.alphas[idx] = alpha
            self.classifier_errors[idx] = classifier_error
            
            if classifier_error == 0:
                break
            
            wsum = np.sum(w)
            
            if wsum <= 0:
                break
            
            if idx < self.n_classifiers - 1:
                w /= wsum
        return self
    
    def makeClassifier(self):
        '''Initialize a copy of the weak classifier'''
        clf = CLF(self.learner)
        return clf.classifier

    def boost(self, idx, X, y, w):
        '''Perform one boosting iteration.
        
        Create a weak classifier, fit it to the data,
        evaluate the error and update weights accordingly.
        '''
        classifier = self.makeClassifier()
        classifier.fit(X, y, sample_weight = w)
        self.classifiers.append(classifier)
        y_pred = classifier.predict(X)
        
        if idx == 0: 
            self.classes = getattr(classifier, 'classes_', None)
            self.n_classes = len(self.classes)
        
        mistakes = y_pred != y
        
        classifier_error = np.mean(np.average(mistakes, weights = w, axis = 0))
        if classifier_error <= 0:
            return w, 1., 0.
        
        n_classes = self.n_classes
        
        if classifier_error >= 1. - (1. / n_classes):
            self.classifiers.pop(-1)
            if len(self.classifiers) == 0:
                print("Error: Could not fit classifier.")
            return None, None, None
        
        alpha = self.learn_rate * (np.log((1. - classifier_error) / classifier_error) + 
                                               np.log(n_classes - 1.))
        if idx != self.n_classifiers - 1:
            w *= np.exp(alpha * mistakes * (w > 0))
            
        return w, alpha, classifier_error
            
    def predict(self, X, y = None):
        '''Predict labels for a training samples. 

        Parameters
        ----------
        X : array (n_test X n_variables)
            Test input for the model
        y : array, optional
            Correct test output labels. When given, the model will automatically
            check its predictions and store the results. The default is None.

        Returns
        -------
        None.

        '''
        pred = self.choose(X)
        self.predictions = self.classes.take(np.argmax(pred, axis = 1), axis = 0)
        if any(y) == True:
            self.check(y)
            
            
    def results(self, return_vals = True, supress_print = True):
        '''Report accuracy and error by returning the values and/or printing.'''
        
        if supress_print != True:
            print(f'Final Accuracy: {self.accuracy:3.3%}')
            print(f'Final Error Rate: {self.error_rate:3.3%}')
        if return_vals == True:
            return self.accuracy, self.classifier_errors
        
        
    def choose(self, X):
        '''Make a prediction from the base classifier.
        

        Parameters
        ----------
        X : array (n_train X n_variables)
            training sample

        Returns
        -------
        pred : output label
            The predicted value 

        '''
        classes = self.classes[:, np.newaxis]
        pred = np.sum((classifier.predict(X) == classes).T * alpha
                   for classifier, alpha in zip(self.classifiers, self.alphas))
        pred /= np.sum(self.alphas)
        return pred
        
        
    def check(self, y):
        self.accuracy = np.sum(np.where(self.predictions == y, 1, 0)) / len(self.predictions)
        self.error_rate = np.sum(np.where(self.predictions != y, 1, 0)) / len(self.predictions)
        
        
        
def compare_learn_rates(X, y, learn_rates, num_cycles, max_depth):
    acc_mat = np.zeros((num_cycles, len(learn_rates)))
    for cycle in range(num_cycles):
        print(f'Starting cycle {cycle}...')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) # resplits the data every cycle to randomize results
        for idx, rate in enumerate(learn_rates):
            
            boost = Adaboost(learner = DecisionTreeClassifier(max_depth = max_depth), n_classifiers = 100, learn_rate = rate)
            boost.fit(X_train, y_train)
            boost.predict(X_test, y_test)
            acc, _ = boost.results()
            acc_mat[cycle, idx] = acc
            #print(f'Learn Rate: {rate} : Accuracy: {acc:.4%}')
    avg_acc = np.mean(acc_mat, axis = 0)
    wins = np.argmax(acc_mat, axis = 1)
    lr_idx, win_cnt = np.unique(wins, return_counts = True)
    win_cnt_dict = dict(zip(lr_idx, win_cnt))
    for idx, rate in enumerate(learn_rates):
        print(f'Learn Rate = {rate}:')
        print(f'Average Accuracy: {avg_acc[idx]:.4%}')
        print(f'Had the best accuracy {win_cnt_dict.get(idx, 0)} times.\n')
            
    return avg_acc

def fix_train_error(error_list):
    fixed_list = [err if err != 1 else 0 for err in error_list]
    return fixed_list

def make_result_figure(dataset_name, X, y, learn_rate = 0.05, max_depth = 10, max_itr = 1800, n_classifiers = 100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    
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
    
    fig.suptitle(f'{dataset_name} Test: Results', x = 0.5, y = 1.03, fontsize = 24)
    fig.tight_layout(pad = 0.5, w_pad = 0.5, h_pad = 0.5)
    
    return fig