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
models = [model1, model2, model3, model5, model6, model7, model8, model9]

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

# Import Custom Chosen Banknote Authentication Data From TXT Found On UCL
banknote_data = np.loadtxt('Datasets/data_banknote_authentication.txt', delimiter = ',')
banknote_X = banknote_data[:,0:4]
banknote_y = banknote_data[:,4]
num_samples = len(banknote_X)
description = 'Set of banknote feature extraction data used for classification of valid and invalid banknotes.'
data_dim5 = []
data_dim5.append('1 Float Input -- Representing: Variance Of Wavelet Transformed Image')
data_dim5.append('1 Float Input -- Representing: Skewness Of Wavelet Transformed Image')
data_dim5.append('1 Float Input -- Representing: Curtosis Of Wavelet Transformed Image')
data_dim5.append('1 Float Input -- Representing: Entropy Of Image')
dataset5 = Dataset('Banknote Authentication', banknote_X, banknote_y, description, 'Binary Value: 0 for Invalid and 1 for Valid', num_samples, data_dim5)

# Import Custom Mushroom Classification Data From CSV Found On Kaggle
mushroom_data = pd.read_csv('Datasets/mushrooms.csv')
data_dim6 = []
mushroom_data['class'] = mushroom_data['class'].replace({'e':0, 'p':1})
mushroom_data['cap-shape'] = mushroom_data['cap-shape'].replace({'b':0, 'c':1, 'x':2, 'f':3, 'k':4, 's':5})
data_dim6.append('1 Integer Input: 0-5 -- Representing: Cap Shape')
mushroom_data['cap-surface'] = mushroom_data['cap-surface'].replace({'f':0, 'g':1, 'y':2, 's':3})
data_dim6.append('1 Integer Input: 0-3 -- Representing: Cap Surface')
mushroom_data['cap-color'] = mushroom_data['cap-color'].replace({'n':0, 'b':1, 'c':2, 'g':3, 'r':4, 'p':5, 'u':6, 'e':7, 'w':8, 'y':9})
data_dim6.append('1 Integer Input: 0-9 -- Representing: Cap Color')
mushroom_data['bruises'] = mushroom_data['bruises'].replace({'t':0, 'f':1})
data_dim6.append('1 Binary Input: 0-1 -- Representing: Bruises')
mushroom_data['odor'] = mushroom_data['odor'].replace({'a':0, 'l':1, 'c':2, 'y':3, 'f':4, 'm':5, 'n':6, 'p':7, 's':8})
data_dim6.append('1 Integer Input: 0-8 -- Representing: Odor')
mushroom_data['gill-attachment'] = mushroom_data['gill-attachment'].replace({'a':0, 'd':1, 'f':2, 'n':3})
data_dim6.append('1 Integer Input: 0-3 -- Representing: Gill Attachement')
mushroom_data['gill-spacing'] = mushroom_data['gill-spacing'].replace({'c':0, 'w':1, 'd':2})
data_dim6.append('1 Integer Input: 0-2 -- Representing: Gill Spacing')
mushroom_data['gill-size'] = mushroom_data['gill-size'].replace({'b':0, 'n':1})
data_dim6.append('1 Binary Input: 0-1 -- Representing: Gill Size')
mushroom_data['gill-color'] = mushroom_data['gill-color'].replace({'k':0, 'n':1, 'b':2, 'h':3, 'g':4, 'r':5, 'o':6, 'p':7, 'u':8, 'e':9, 'w':10, 'y':11})
data_dim6.append('1 Integer Input: 0-12 -- Representing: Gill Color')
mushroom_data['stalk-shape'] = mushroom_data['stalk-shape'].replace({'t':0, 'e':1})
data_dim6.append('1 Binary Input: 0-1 -- Representing: Stalk Shape')
mushroom_data['stalk-root'] = mushroom_data['stalk-root'].replace({'?':0, 'b':1, 'e':2, 'c':3, 'r':4})
data_dim6.append('1 Integer Input: 0-4 -- Representing: Stalk Root')
mushroom_data['stalk-surface-above-ring'] = mushroom_data['stalk-surface-above-ring'].replace({'y':0, 's':1, 'k':2, 'f':3})
data_dim6.append('1 Integer Input: 0-3 -- Representing: Stalk Surface Above Ring')
mushroom_data['stalk-surface-below-ring'] = mushroom_data['stalk-surface-below-ring'].replace({'y':0, 's':1, 'k':2, 'f':3})
data_dim6.append('1 Integer Input: 0-3 -- Representing: Stalk Surface Below Ring')
mushroom_data['stalk-color-above-ring'] = mushroom_data['stalk-color-above-ring'].replace({'n':0, 'b':1, 'c':2, 'g':3, 'o':4, 'p':5, 'e':6, 'w':7, 'y':8})
data_dim6.append('1 Integer Input: 0-8 -- Representing: Stalk Color Above Ring')
mushroom_data['stalk-color-below-ring'] = mushroom_data['stalk-color-below-ring'].replace({'n':0, 'b':1, 'c':2, 'g':3, 'o':4, 'p':5, 'e':6, 'w':7, 'y':8})
data_dim6.append('1 Integer Input: 0-8 -- Representing: Stalk Color Below Ring')
mushroom_data['veil-type'] = mushroom_data['veil-type'].replace({'p':0, 'u':1})
data_dim6.append('1 Binary Input: 0-1 -- Representing: Veil Type')
mushroom_data['veil-color'] = mushroom_data['veil-color'].replace({'n':0, 'o':1, 'w':2, 'y':3})
data_dim6.append('1 Integer Input: 0-3 -- Representing: Veil Color')
mushroom_data['ring-number'] = mushroom_data['ring-number'].replace({'n':0, 'o':1, 't':2})
data_dim6.append('1 Integer Input: 0-2 -- Representing: Ring Number')
mushroom_data['ring-type'] = mushroom_data['ring-type'].replace({'c':0, 'e':1, 'f':2, 'l':3, 'n':4, 'p':5, 's':6, 'z':7})
data_dim6.append('1 Integer Input: 0-7 -- Representing: Ring Type')
mushroom_data['spore-print-color'] = mushroom_data['spore-print-color'].replace({'k':0, 'n':1, 'b':2, 'h':3, 'r':4, 'o':5, 'u':6, 'w':7, 'y':8})
data_dim6.append('1 Integer Input: 0-8 -- Representing: Spore Print Color')
mushroom_data['population'] = mushroom_data['population'].replace({'a':0, 'c':1, 'n':2, 's':3, 'v':4, 'y':5})
data_dim6.append('1 Integer Input: 0-5 -- Representing: Population')
mushroom_data['habitat'] = mushroom_data['habitat'].replace({'g':0, 'l':1, 'm':2, 'p':3, 'u':4, 'w':5, 'd':6})
data_dim6.append('1 Integer Input: 0-6 -- Representing: Habitat')
mushroom_X = mushroom_data.drop(columns = 'class')
mushroom_y = mushroom_data['class']
num_samples = len(mushroom_X)
description = 'Set of categorical data about properties which is converted into numeric data for use in classification of edible mushrooms against poisonous ones.'
dataset6 = Dataset('Mushroom Classification', mushroom_X, mushroom_y, description, 'Binary Value: 0 for Edible and 1 for Poisonous', num_samples, data_dim6)

# Import Custom Glass Classification Data From CSV Found On Kaggle
glass_data = pd.read_csv('Datasets/glass.csv')
glass_X = glass_data.drop(columns = 'Type')
glass_y = glass_data['Type']
num_samples = len(glass_X)
description = 'Set of numerical data about chemical properties of glass used for classification of glass type.'
data_dim7 = []
data_dim7.append('1 Float Input -- Representing: Refractive Index')
data_dim7.append('1 Float Input -- Representing: Sodium Content')
data_dim7.append('1 Float Input -- Representing: Magnesium Content')
data_dim7.append('1 Float Input -- Representing: Aluminum Content')
data_dim7.append('1 Float Input -- Representing: Silicon Content')
data_dim7.append('1 Float Input -- Representing: Potassium Content')
data_dim7.append('1 Float Input -- Representing: Calcium Content')
data_dim7.append('1 Float Input -- Representing: Barium Content')
data_dim7.append('1 Float Input -- Representing: Iron Content')
dataset7 = Dataset('Glass Classification', glass_X, glass_y, description, 'Integer Value: 1-7', num_samples, data_dim7)

# Import Custom NASA Asteroid Classification Data From CSV Found On Kaggle
asteroid_data = pd.read_csv('Datasets/nasa.csv')
asteroid_X = asteroid_data.drop(columns = ['Neo Reference ID', 'Name', 'Orbiting Body', 'Equinox', 'Hazardous', 'Orbit Determination Date', 'Close Approach Date'])
asteroid_y = asteroid_data['Hazardous']
num_samples = len(asteroid_X)
description = 'Set of information about asteroids for classification of them being hazardous or not'
data_dim8 = []
data_dim8.append('33 Float Inputs -- Representings: Various Asteroid Features')
dataset8 = Dataset('NASA: Asteroids Classification', asteroid_X, asteroid_y, description, 'Binary Value: 0 for Non-Hazardous and 1 for Hazardous', num_samples, data_dim8)

# Import Custom Red Wine Quality Classification Data From CSV Found On Kaggle
wine_data = pd.read_csv('Datasets/winequality-red.csv')
wine_X = wine_data.drop(columns = 'quality')
wine_y = wine_data['quality']
num_samples = len(wine_X)
description = 'Set of numerical data used for classification of the quality of red wines.'
data_dim9 = []
data_dim9.append('1 Float Input -- Representing: Fixed Acidity')
data_dim9.append('1 Float Input -- Representing: Volatile Acidity')
data_dim9.append('1 Float Input -- Representing: Citric Acid')
data_dim9.append('1 Float Input -- Representing: Residual Sugar')
data_dim9.append('1 Float Input -- Representing: Chlorides')
data_dim9.append('1 Float Input -- Representing: Free Sulfur Dioxide')
data_dim9.append('1 Float Input -- Representing: Total Sulfur Dioxide')
data_dim9.append('1 Float Input -- Representing: Density')
data_dim9.append('1 Float Input -- Representing: pH')
data_dim9.append('1 Float Input -- Representing: Sulphates')
data_dim9.append('1 Float Input -- Representing: Alcohol')
dataset9 = Dataset('Red Wine Quality', wine_X, wine_y, description, 'Integer Values: 0-10', num_samples, data_dim9)

# Import Heart Disease Classification Data From CSV Found On Kaggle
heart_data = pd.read_csv('Datasets/heart.csv')
heart_X = heart_data.drop(columns = 'target')
heart_y = heart_data['target']
num_samples = len(heart_X)
description = 'Set of numerical data used for classification of the presence of heart disease.'
data_dim10 = []
data_dim10.append('1 Integer Input -- Representing: Age')
data_dim10.append('1 Binary Input: 0-1 -- Representing: Sex')
data_dim10.append('1 Integer Input -- Representing: Chest Pain Type')
data_dim10.append('1 Integer Input -- Representing: Resting Blood Pressure')
data_dim10.append('1 Integer Input -- Representing: Serum Cholestoral')
data_dim10.append('1 Binary Input: 0-1 -- Representing: Fasting Blood Sugar')
data_dim10.append('1 Binary Input: 0-1 -- Representing: Resting Electrocardiographic Results')
data_dim10.append('1 Integer Input -- Representing: Maximum Heart Rate Achieved')
data_dim10.append('1 Binary Input: 0-1 -- Representing: Exercise Induced Angina')
data_dim10.append('1 Integer Input -- Representing: ST Depression Induced By Exercise')
data_dim10.append('1 Integer Input -- Representing: Slope Of The Peak Exercise ST Segment')
data_dim10.append('1 Integer Input -- Representing: Number Of Major Vessels')
data_dim10.append('1 Integer Input -- Representing: Thal')
dataset10 = Dataset('Heart Disease', heart_X, heart_y, description, 'Binary Value: 0 for No Presence and 1 for Presence', num_samples, data_dim10)

# Store the imported datasets into the dataset array
data_sets = [dataset1, dataset2, dataset3, dataset4, dataset5, dataset6, dataset7, dataset8 ,dataset9, dataset10]

# Run the model comparisons on the given datasets and print the results
print('Beginning Model Comparisons...')
print()
results = compare_models(data_sets, models)
print(results.head(20))
print(results.tail(20))
