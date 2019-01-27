import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.simplefilter('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Imputer

# Loading the data set
train_data = pd.read_csv("Data/train.csv")
test_data = pd.read_csv("Data/test.csv")


# Selecting features for training the model
input_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_X = train_data[input_features]
train_y = train_data['Survived']

# Selecting features for testing the model on new data
test_X = test_data[input_features]


'''
Missing values
    |__ Training Data
        |__ train_X
            |__ Age:        177 missing values
            |__ Embarked:   2 missing values
        |__ train_y
            |__ 0 missing values
    |__ Testing Data
        |__ test_X
            |__ Age:        86 missing values
            |__ Fare:       1 missing value(s)
'''

train_X['Age'].fillna(train_X['Age'].mean(), inplace=True)  # Filling missing values with median 'Age' values
train_X['Embarked'].fillna(method='ffill', inplace=True)    # Filling missing values with "Forward Fill Method"
test_X['Age'].fillna(test_X['Age'].mean(), inplace=True)    # Filling missing values with median 'Age' values
test_X['Fare'].fillna(test_X['Fare'].mean(), inplace=True)  # Filling missing values with median 'Fare' value(s)


'''
Shape of Training and Testing Data
train_X:    (891 x 7)
train_y:    (891 x 1)
test_X:     (418 x 7)
prediction: (418 x 1)
'''

# Encoding Text to numerical values
label_encoder = LabelEncoder()
for col in train_X:
    if train_X[col].dtype == 'O':
        train_X[col] = label_encoder.fit_transform(train_X[col])
    else:
       pass
for col in test_X:
    if test_X[col].dtype == 'O':
        test_X[col] = label_encoder.fit_transform(test_X[col])
    else:
        pass


'''
To-Do
1. Select appropriate models
2. Fit the model and train it using Training Data
3. Predict the values on Test Data
'''

'''
# Training the algorithm on training set
model = LogisticRegression()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print(prediction)
'''


