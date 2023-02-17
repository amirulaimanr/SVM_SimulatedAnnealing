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
headers = ["id","clump_thickness","uniformity_of_cell_size","uniformity_of_cell_shape","marginal_adhesion","single_epithelial_cell_size","bare_nuclei","bland_chromatin","normal_nucleoli","mitoses","class"]
data = pd.read_csv('breast-cancer-wisconsin.data', delimiter=',', header=None, names = headers)

# Step 3: Handling missing values
data = data.replace('?', pd.np.nan)
data = data.dropna()

# Step 4: Split the data into feature and target variables
X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]

# Step 5: Handling the categorical data
le = LabelEncoder()
y = le.fit_transform(y)

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

# Print precision, fscore, recall, and accuracy
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F-score:", f1_score(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

