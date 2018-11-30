import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

titanic_data = pd.read_csv("/Users/akshaybhaskaran/PycharmProjects/ML/TitanicPrediction/Data/train.csv")

input_features = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
training_X = titanic_data[input_features]

print(training_X)






