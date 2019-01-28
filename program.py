import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.simplefilter('ignore')
from keras.models import Sequential
from keras.layers import Dense

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

# Selecting the model
model = Sequential()

# Input layer
model.add(Dense(output_dim=14, activation='relu',input_dim=7))

# Hidden layers
model.add(Dense(14, activation='relu'))

# Output layer
model.add(Dense(1, activation='sigmoid'))

# Compile Model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(train_X, train_y, epochs=200)

# Model Score
scores = model.evaluate(train_X, train_y)
print("\n%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

# Making prediction
prediction = model.predict_classes(test_X)
print(prediction.shape)

# Writing predictions to a csv file - for submission
submission_data = pd.DataFrame(test_data['PassengerId'])
submission_data['Survived'] = prediction
submission_data.to_csv('submissions.csv', sep=",", index=False)
