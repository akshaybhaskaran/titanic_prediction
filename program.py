import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.simplefilter('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer


titanic_data = pd.read_csv("/Users/akshaybhaskaran/PycharmProjects/ML/TitanicPrediction/Data/train.csv")

input_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
training_X = titanic_data[input_features]

print(training_X.shape)
print(training_X.isnull().sum())

# Encoding the 'Sex' and 'Embarked' Columns to base everything with Integers
label_enc = LabelEncoder()
training_X['Sex'] = label_enc.fit_transform(training_X['Sex'])
training_X['Embarked'] = label_enc.fit_transform(training_X['Embarked'])

# Working on filling the missing values for 'Age' using Imputer
'''write code for working on the missing values of Age column'''