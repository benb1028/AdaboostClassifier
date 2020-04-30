# Author: Ben Brooks and Ryan Marshall
# University: Clarkson University
#
# Created For: CS449 Final Project
#
# Model Comparison
#
# Imported Common Libraries
import pandas as pd
import numpy as np
# Imported Machine Learning Classes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# compare_models() Function: 
# Compares the differents set of datasets against the different sets of models over the number of iterations, using a set number of classifiers. 
# The datasets are randomly split for each iterations into some random assortment of training and testing data.
#
# Inputs:
# data_sets - An array of out custom dataset class, defining the datasets to be used
# models - An array of our custom model class, defining the models to be used
# model_itrs - The number of iterations to use - Default: 5
# n_classifiers - The number of classifiers to use - Default: 100
#
# Outputs:
# pd.DataFrame(results) - A pandas dataframe containing the results of the model comparisons
#
def compare_models(data_sets, models, model_itrs = 5, n_classifiers = 100):
    # Create an array to contain the results of the model comparison as well as an index to print the dataset number
    results = []
    idxD = 0
    # Loop through all of the datasets
    for dataset in data_sets:
        # Store the values of the dataset inputs and outputs
        X = dataset.data_input
        y = dataset.data_output
        # Print the current dataset index as well as the information about the current dataset
        print(f'Starting Dataset #: {idxD + 1} / {len(data_sets)}')
        print(f'Dataset Name: {dataset.name}')
        print(f'Dataset Description: {dataset.desc}')
        print(f'Dataset Output Format: {dataset.out_format}')
        print(f'Dataset Samples Amount: {dataset.samp_num}')
        # Loop through the datasets input dimension formats to print their information
        for z in range(len(dataset.dim_format)):
            print(f'Dataset Dimensional Format: {dataset.dim_format[z]}')
        # Increment the dataset index and store all necessary variables, while also creating a model index
        idxD += 1
        n_samples, n_variables = X.shape
        classes, counts = np.unique(y, return_counts = True)
        n_out_classes = len(classes)
        idxM = 0
        # Loop through all of the models for this particular dataset
        for model in models:
            # Print the current model index as well as the information about the current model
            print(f'    Starting Model #: {idxM + 1} / {len(models)}')
            print(f'    Model Name: {model.name}')
            print(f'    Model Algorithm: {model.algorithm}')
            idxM += 1
            # Increment the model index and begin looping through the specified number of iterations
            for itr in range(model_itrs):
                # Print the current iteration that the given model is training on for the given dataset
                print(f'        Starting iteration {itr + 1} / {model_itrs}...')
                # Create a random split of the data that is used as input to create a random combination of testing and training data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
                # Store some variables into the result array
                result = {}
                result['sample_size'] = n_samples
                result['num_variables'] = n_variables
                result['num_classes'] = n_out_classes
                result['model'] = model.name
                # Run the AdaBoostClassifier on the given model for the given dataset
                classifier = AdaBoostClassifier(model.model,
                                                learning_rate = 0.05,
                                                n_estimators = n_classifiers, algorithm = model.algorithm).fit(X_train, y_train)
                # Store the results of the AdaBoostClassifier into the result array
                result['accuracy'] = classifier.score(X_test, y_test)
                # Append this iterations result to the overall results array
                results.append(result)
    # Return the results array of the model comparisons on the given datasets and models
    return(pd.DataFrame(results))
           
# Model() Class:
# Class containing some useful information about the models chosen for the model comparison.
#
# Initializers:
# name - The string name of the model being used
# model - The sklearn model being used
# algorithm - The string of the sklearn AdaBoost algorithm to use
#
class Model():
    def __init__(self, name, model, algorithm = "SAMME.R"):
        self.name = name
        self.model = model
        self.algorithm = algorithm

# Dataset() Class:
# Class containing some useful information about the datasets chosen for the model comparison.
#
# Initializers:
# name - The string name of the dataset being used
# data_input - The dataframe of the dataset input being used
# data_output - The dataframe of the dataset output being used
# desc - The string description of the dataset being used
# out_format - The string name of the output format of the data
# samp_num - The integer value of the number of samples of data
# dim_format - The array of strings describing the properties of each samples dimensions
#
class Dataset():
    def __init__(self, name, data_input, data_output, desc, out_format, samp_num, dim_format):
        self.name = name
        self.data_input = data_input
        self.data_output = data_output
        self.desc = desc
        self.out_format = out_format
        self.samp_num = samp_num
        self.dim_format = dim_format
        #Add # binary dimensions
        #Add # categorical dimensions
        #Add # output classes