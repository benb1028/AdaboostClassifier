# Author: Ben Brooks

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression



def compare_models(datasets, models, model_itrs = 10, n_classifiers = 100):
    results = []
    for X, y in datasets:
        
        n_samples, n_variables = X.shape
        classes, counts = np.unique(y, return_counts = True)
        n_out_classes = len(classes)
        for model in models:
            for itr in range(model_itrs):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
                result = {}
                result['sample_size'] = n_samples
                result['num_variables'] = n_variables
                result['num_classes'] = n_out_classes
                result['model'] = model.name
                classifier = AdaBoostClassifier(model.model,
                                                learning_rate = 0.05,
                                                n_estimators = n_classifiers).fit(X_train,
                                                                                  y_train)
                
                result['accuracy'] = classifier.score(X_test, y_test)
                results.append(result)
    return(pd.DataFrame(results))
                
class Model():
    def __init__(self, name, model):
        self.name = name
        self.model = model