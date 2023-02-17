# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, precision_score, f1_score, recall_score, \
    accuracy_score

# Import necessary libraries
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Import the dataset
headers = ["Class","Alcohol","Malic acid","Ash","Alcalinity of ash","Magnesium","Total phenols","Flavanoids","Nonflavanoid phenols","Proanthocyanins","Color intensity","Hue","OD280/OD315 of diluted wines","Proline"]
data = pd.read_csv('wine.data', delimiter=',', header=None, names = headers)

# Step 3: Split the data into feature and target variables
X = data.iloc[:, 1:]
y = data.iloc[:, 0]

# Step 4: Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create an instance of the SVM classifier
clf = SVC()

# Train the model on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Print the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print(classification_report(y_test, y_pred))

from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score

# Calculate precision
precision = precision_score(y_test, y_pred, average='weighted')
print("Precision: {:.2f}%".format(precision*100))

# Calculate f1-score
f1 = f1_score(y_test, y_pred, average='weighted')
print("F1-Score: {:.2f}%".format(f1*100))

# Calculate recall
recall = recall_score(y_test, y_pred, average='weighted')
print("Recall: {:.2f}%".format(recall*100))

# Calculate accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(acc*100))

