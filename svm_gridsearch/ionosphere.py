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

# Import the dataset
data = pd.read_csv('ionosphere.data', header=None)

# Split the data into feature and target variables
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Handling the categorical data
le = LabelEncoder()
y = le.fit_transform(y)

# Step 4: Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Set the parameter grid for grid search
C_range = np.arange(0.1, 10.1, 0.1)
gamma_range = np.arange(0.001, 1.001, 0.001)

param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)

# Create the grid search object
grid_search = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)

# Fit the grid search object to the training data
grid_search.fit(X_train, y_train)

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Print the best parameters
print(grid_search.best_params_)

print(grid_search.best_estimator_)
# best_estimator_ is the model with the best parameters
best_model = grid_search.best_estimator_

# Generate predictions for test dataset
y_pred = best_model.predict(X_test)

# Calculate precision, recall, F1 score, and accuracy
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1)
print("Accuracy: ", accuracy)

# Concatenate the grid search results into a single dataframe
df_combinations = pd.concat([pd.DataFrame(grid_search.cv_results_["params"]), pd.DataFrame(grid_search.cv_results_["mean_test_score"], columns=["mean_test_score"])], axis=1)

df_combinations.to_csv('ionosphere_grid.csv')

