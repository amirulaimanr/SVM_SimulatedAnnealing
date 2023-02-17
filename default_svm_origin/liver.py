# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, precision_score, f1_score, recall_score, \
    accuracy_score

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Step 1: Import the dataset
headers = ['mcv', 'alkphos', 'sgpt', 'sgot', 'gammagt', 'drinks', 'selector']
data = pd.read_csv("bupa.data", names=headers)

# Step 3: Handling missing values
data = data.replace('?', pd.np.nan)
data = data.dropna()

# Step 4: Converting categorical features into numerical values
le = LabelEncoder()
data['selector'] = le.fit_transform(data['selector'])

X = data.drop('selector', axis=1)
y = data['selector']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

##Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC()

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

print(classification_report(y_test, y_pred))

# Print precision, fscore, recall, and accuracy
print("Precision:", precision_score(y_test, y_pred))
print("F-score:", f1_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

