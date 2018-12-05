import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.simplefilter('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Imputer


train_data = pd.read_csv("Data/train.csv")
test_data = pd.read_csv("Data/test.csv")

input_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_X = train_data[input_features]
test_X = test_data[input_features]
train_y = train_data['Survived']
test_y = train_y
'''for now we can assign our prediction target (test_y) to be equal to our given/training predictions (train_y)'''



# Encoding the 'Sex' and 'Embarked' Columns to base everything with Integers
label_enc = LabelEncoder()
train_X['Sex'] = label_enc.fit_transform(train_X['Sex'])
test_X['Sex'] = label_enc.fit_transform(test_X['Sex'])

train_X['Embarked'] = label_enc.fit_transform(train_X['Embarked'])
test_X['Embarked'] = label_enc.fit_transform(test_X['Embarked'])

# Filling the missing values in 'Age' column, using Pandas fillna
train_X['Age'].fillna(train_X['Age'].mean(), inplace=True)
test_X['Age'].fillna(test_X['Age'].mean(), inplace=True)
test_X['Fare'].fillna(test_X['Fare'].mean(), inplace=True)


# Training the algorithm on training set
model = LogisticRegression()
model.fit(train_X, train_y)
prediction = model.predict(test_X)

'''make the model more generic instead of directly giving the column names
tweak the model's default parameters to see a change in predictions
find out the accuracy of the classifier'''