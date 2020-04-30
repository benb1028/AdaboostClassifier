# Author: Ben Brooks
# Date of Creation:
import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

from model_comparison import Model
from model_comparison import compare_models


model1 = Model('Decision Tree - Depth 1', DecisionTreeClassifier(max_depth = 1))
model2 = Model('Decision Tree - Depth 5', DecisionTreeClassifier(max_depth = 5))
model3 = Model('Decision Tree - Depth 10', DecisionTreeClassifier(max_depth = 10))
model4 = Model('Logistic Regression', LogisticRegression(max_iter = 1800))
model5 = Model('Random Forest Classifier - Depth 1', RandomForestClassifier(max_depth = 1))
model6 = Model('Random Forest Classifier - Depth 5', RandomForestClassifier(max_depth = 5))
model7 = Model('Random Forest Classifier - Depth 10', RandomForestClassifier(max_depth = 10))
model8 = Model('Stochastic Gradient Descent', SGDClassifier(), alg = "SAMME")
model9 = Model('SVC', SVC(), alg = "SAMME")



dataset_list = []
models = [model1, model2, model3, model4, model5, model6, model7, model8, model9]

digits = datasets.load_digits()
dataset_list.append((digits['data'], digits['target']))


banknote_data = np.loadtxt('data_banknote_authentication.txt', delimiter = ',')
dataset_list.append((banknote_data[:,0:4], banknote_data[:,4]))
'''
clothing_data = pd.read_csv('fashion-mnist_train.csv')
clothing_X = clothing_data.drop(columns = 'label')
clothing_y = clothing_data['label']
dataset_list.append((clothing_X, clothing_y))
'''
titanic_data = pd.read_csv('titanic_train.csv')
titanic_data = titanic_data.drop(columns = ['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin', 'Embarked']).dropna()
titanic_data['Sex'] = titanic_data['Sex'].replace({'male':0, 'female':1})
titanic_X = titanic_data.drop(columns = 'Survived')
titanic_y = titanic_data['Survived']
dataset_list.append((titanic_X, titanic_y))
'''
# replace categorical data with numbers
mushroom_data = pd.read_csv('mushrooms.csv')
mushroom_X = mushroom_data.drop(columns = 'class')
mushroom_y = mushroom_data['class']
dataset_list.append((mushroom_X, mushroom_y))
'''
glass_data = pd.read_csv('glass.csv')
glass_X = glass_data.drop(columns = 'Type')
glass_y = glass_data['Type']
dataset_list.append((glass_X, glass_y))

asteroid_data = pd.read_csv('nasa.csv')
asteroid_X = asteroid_data.drop(columns = ['Neo Reference ID', 'Name', 'Orbiting Body', 'Equinox', 'Hazardous', 'Orbit Determination Date', 'Close Approach Date'])
asteroid_y = asteroid_data['Hazardous']
dataset_list.append((asteroid_X, asteroid_y))

wine_data = pd.read_csv('winequality-red.csv')
wine_X = wine_data.drop(columns = 'quality')
wine_y = wine_data['quality']
dataset_list.append((wine_X, wine_y))

heart_data = pd.read_csv('heart.csv')
heart_X = heart_data.drop(columns = 'target')
heart_y = heart_data['target']
dataset_list.append((heart_X, heart_y))

cancer_data = datasets.load_breast_cancer()
cancer_X = cancer_data['data']
cancer_y = cancer_data['target']
dataset_list.append((cancer_X, cancer_y))





results = compare_models(dataset_list, models)
print(results.head(20))
print(results.tail(20))
