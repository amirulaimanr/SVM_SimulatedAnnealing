# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, precision_score, f1_score, recall_score, \
    accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Step 1: Import the dataset
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


# split the dataset into 70% training set and 30% test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# split the training set into 70% train and 30% validation
X_train2, X_val, y_train2, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=0)

# Set the parameter grid for grid search
C_range = np.arange(0.1, 10.1, 0.1)
gamma_range = np.arange(0.001, 1.001, 0.001)

param_grid = dict(gamma=gamma_range, C=C_range)

# create a SVM classifier
clf = svm.SVC()

# perform grid search
grid_search = GridSearchCV(clf, param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_train2, y_train2)

# get the best parameters
best_params = grid_search.best_params_
print(best_params)

# retrain the model with the best parameters on the validation set
clf = svm.SVC(C=best_params['C'], gamma=best_params['gamma'])
clf.fit(X_val, y_val)

# predict accuracy on the test set
accuracy = clf.score(X_test, y_test)
print("Accuracy: {:.2f}%".format(accuracy*100))


#Predict the response for test dataset
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

# Print precision, fscore, recall, and accuracy
print("Precision:", precision_score(y_test, y_pred))
print("F-score:", f1_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))


# Concatenate the grid search results into a single dataframe
df_combinations = pd.concat([pd.DataFrame(grid_search.cv_results_["params"]), pd.DataFrame(grid_search.cv_results_["mean_test_score"], columns=["mean_test_score"])], axis=1)

df_combinations.to_csv('liver3_grid.csv')


