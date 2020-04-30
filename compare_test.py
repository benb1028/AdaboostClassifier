# Authors: Ben Brooks and Ryan Marshall
# University: Clarkson University
#
# Created For: CS449 Final Project
#
# Data Importation And Model Creation
#
# Imported Common Libraries
import pandas as pd
import numpy as np
# Imported Machine Learning Classes
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
# Imported Custom Classes
from model_comparison import Model
from model_comparison import Dataset
from model_comparison import compare_models

# Print that models are being created
print('Models are being created...')

# Define the models to be used in the model comparison on the chosen datasets
model1 = Model('Decision Tree - Depth 1', DecisionTreeClassifier(max_depth = 1))
model2 = Model('Decision Tree - Depth 5', DecisionTreeClassifier(max_depth = 5))
model3 = Model('Decision Tree - Depth 10', DecisionTreeClassifier(max_depth = 10))
model4 = Model('Logistic Regression', LogisticRegression(max_iter = 1800))
model5 = Model('Random Forest Classifier - Depth 1', RandomForestClassifier(max_depth = 1))
model6 = Model('Random Forest Classifier - Depth 5', RandomForestClassifier(max_depth = 5))
model7 = Model('Random Forest Classifier - Depth 10', RandomForestClassifier(max_depth = 10))
model8 = Model('Stochastic Gradient Descent', SGDClassifier(), algorithm = "SAMME")
model9 = Model('SVC', SVC(), algorithm = "SAMME")
#models = [model1, model2, model3, model4, model5, model6, model7, model8, model9]
models = [model1, model2]

# Print that datasets are being imported
print('Datasets are being imported...')

# Import SKLearn Built In Dataset MNIST Digits Dataset
digits = datasets.load_digits()
num_samples = len(digits['data'])
description = 'Set of images of handwritten numbers used for classification of the specific number given in an image.'
data_dim1 = []
data_dim1.append('64 Integer Inputs: 0-16 -- Representing: Pixel Value')
dataset1 = Dataset('MNIST', digits['data'], digits['target'], description, 'Integer Value: 0-9', num_samples, data_dim1)

# Import SKLearn Built In Breast Cancer Dataset
cancer_data = datasets.load_breast_cancer()
num_samples = len(cancer_data['data'])
description = 'Set of attributes of a cell nuclei used for classification of malignant or benign tumors.'
data_dim2 = []
data_dim2.append('3 Float Inputs -- Representing: Nuclei Radius')
data_dim2.append('3 Float Inputs -- Representing: Nuclei Texture')
data_dim2.append('3 Float Inputs -- Representing: Nuclei Perimeter')
data_dim2.append('3 Float Inputs -- Representing: Nuclei Area')
data_dim2.append('3 Float Inputs -- Representing: Nuclei Smoothness')
data_dim2.append('3 Float Inputs -- Representing: Nuclei Compactness')
data_dim2.append('3 Float Inputs -- Representing: Nuclei Concavity')
data_dim2.append('3 Float Inputs -- Representing: Nuclei Concave Points')
data_dim2.append('3 Float Inputs -- Representing: Nuclei Symmetry')
data_dim2.append('3 Float Inputs -- Representing: Nuclei Fractal Dimension')
dataset2 = Dataset('Wisconsin Breast Cancer', cancer_data['data'], cancer_data['target'], description, 'Binary Value: 0 for Malignant and 1 for Benign', num_samples, data_dim2)

# Import Fashion-MNIST From CSV As Not Included In SKlearn But Keras
clothing_data1 = pd.read_csv('Datasets/fashion-mnist_train.csv')
clothing_data2 = pd.read_csv('Datasets/fashion-mnist_test.csv')
clothing = [clothing_data1, clothing_data2]
clothing_data = pd.concat(clothing)
clothing_X = clothing_data.drop(columns = 'label')
clothing_y = clothing_data['label']
num_samples = len(clothing_X)
description = 'Set of images of clothing items used for classification of the specific clothing item given in an image.'
data_dim3 = []
data_dim3.append('784 Integer Inputs: 0-255 -- Representing: Pixel Value')
dataset3 = Dataset('Fashion-MNIST', clothing_X, clothing_y, description, 'Integer Value: 0-9', num_samples, data_dim3)

# Import Titanic From CSV As Not Included In Sklearn But Keras
titanic_data = pd.read_csv('Datasets/titanic_train.csv')
titanic_data = titanic_data.drop(columns = ['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin', 'Embarked']).dropna()
titanic_data['Sex'] = titanic_data['Sex'].replace({'male':0, 'female':1})
titanic_X = titanic_data.drop(columns = 'Survived')
titanic_y = titanic_data['Survived']
num_samples = len(titanic_X)
description = 'Set of information about passengers on the titanic used for classification of their survival.'
data_dim4 = []
data_dim4.append('1 Integer Input: 1-3 -- Representing: Ticket Class')
data_dim4.append('1 Binary Input: 0-1 -- Representing: Sex')
data_dim4.append('1 Integer Input -- Representing: Age')
data_dim4.append('1 Integer Input -- Representing: Number Of Siblings/Spouses Aboard')
data_dim4.append('1 Integer Input -- Representing: Number of Parents/Children Aboard')
dataset4 = Dataset('Titanic', titanic_X, titanic_y, description, 'Binary Value: 0 for Died and 1 for Survived', num_samples, data_dim4)

# Import Custom Chosen Banknote Authentication Data Found On Kaggle
#banknote_data = np.loadtxt('data_banknote_authentication.txt', delimiter = ',')
#dataset_list.append((banknote_data[:,0:4], banknote_data[:,4]))

# Import Custom Mushroom Classification Data Found On Kaggle
# replace categorical data with numbers
#mushroom_data = pd.read_csv('mushrooms.csv')
#mushroom_X = mushroom_data.drop(columns = 'class')
#mushroom_y = mushroom_data['class']
#dataset_list.append((mushroom_X, mushroom_y))

# Import Custom Glass Classification Data Found On Kaggle
#glass_data = pd.read_csv('glass.csv')
#glass_X = glass_data.drop(columns = 'Type')
#glass_y = glass_data['Type']
#dataset_list.append((glass_X, glass_y))

# Import Custom NASA Asteroid Classification Data Found On Kaggle
#asteroid_data = pd.read_csv('nasa.csv')
#asteroid_X = asteroid_data.drop(columns = ['Neo Reference ID', 'Name', 'Orbiting Body', 'Equinox', 'Hazardous', 'Orbit Determination Date', 'Close Approach Date'])
#asteroid_y = asteroid_data['Hazardous']
#dataset_list.append((asteroid_X, asteroid_y))

# Import Custom Red Wine Quality Classification Data Found On Kaggle
#wine_data = pd.read_csv('winequality-red.csv')
#wine_X = wine_data.drop(columns = 'quality')
#wine_y = wine_data['quality']
#dataset_list.append((wine_X, wine_y))

# Import Heart Disease Classification Data Found On Kaggle
#heart_data = pd.read_csv('heart.csv')
#heart_X = heart_data.drop(columns = 'target')
#heart_y = heart_data['target']
#dataset_list.append((heart_X, heart_y))

# Store the imported datasets into the dataset array
data_sets = [dataset4]

# Run the model comparisons on the given datasets and print the results
print('Beginning Model Comparisons...')
print()
results = compare_models(data_sets, models)
print(results.head(20))
print(results.tail(20))
