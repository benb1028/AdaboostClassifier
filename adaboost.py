# Authors: Ben Brooks and Ryan Marshall
# Created for CS449 Assignment 3 at Clarkson University 
# Adaboost Implementation

import numpy as np
from sklearn.tree import DecisionTreeClassifier


stump = DecisionTreeClassifier(max_depth = 1)

class CLF:
    '''Generic classifier to copy'''
    
    def __init__(self, learner):
        self.classifier = learner
        
        
class Adaboost:
    '''Define the class used for boosting.
    
    Contains any methods neccessary for the algorithm and stores the 
    relevent data for analysis and prediction.
    '''
    
    def __init__(self, learner, n_classifiers = 20, learn_rate = 1):
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