import csv
import random
import math
import pandas as pd
import numpy as np

from sklearn import svm, metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from numpy import asarray
from numpy import exp
from numpy.random import randn
from numpy.random import rand
from numpy.random import seed
from matplotlib import pyplot, pyplot as plt
from sklearn.metrics import accuracy_score

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


# Read the GridSearchCV results from a CSV file
results_df = pd.read_csv('liver_grid.csv', names = ["C","gamma","mean_test_score"])

# Extract the C, gamma, and accuracy values from the DataFrame
C = results_df['C'].values
gamma = results_df['gamma'].values
acc_values = results_df['mean_test_score'].values

# Define the objective function for the SVM
def objective(x):
    # split the dataset into 70% training set and 30% test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    C, gamma = x[0], x[1]
    svm = SVC(C=C, gamma=gamma)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    return score

# simulated annealing algorithm
def simulated_annealing(objective, bounds, n_iterations, step_size, temp):
    # generate an initial point
    best = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    print(best)
    best_eval = objective(best)
    # current working solution
    curr, curr_eval = best, best_eval
    scores = list()
    # run the algorithm
    for i in range(n_iterations):
        # take a step
        candidate = curr + randn(len(bounds)) * step_size
        # evaluate candidate point
        candidate_eval = objective(candidate)
        # check for new best solution
        if candidate_eval > best_eval:
            # store new best point
            best, best_eval = candidate, candidate_eval
            # keep track of scores
            scores.append(best_eval)
            # report progress
            print('>%d f(%s) = %.5f' % (i, best, best_eval))
        # difference between candidate and current point evaluation
        diff = candidate_eval - curr_eval
        # calculate temperature for current epoch
        t = temp / float(i + 1)
        # calculate metropolis acceptance criterion
        metropolis = exp(-diff / t)
        # check if we should keep the new point
        if diff < 0 or rand() < metropolis:
            # store the new current point
            curr, curr_eval = candidate, candidate_eval
    return [best, best_eval, scores]

# seed the pseudorandom number generator
seed(1)
# define range for input
bounds = asarray([[0.1, 10], [0.001, 1]])
# define the total iterations
n_iterations = 1000
# define the maximum step size
step_size = asarray([0.1, 0.001])
# initial temperature
temp = 10
# X and y are the data that you have
# perform the simulated annealing search
best, score, scores = simulated_annealing(objective, bounds, n_iterations, step_size, temp)
print('Done!')
print('f(%s) = %f' % (best, score))

#printing the best value
print('\nThe best parameters:')
print(f'Best C: {best[0]}')
print(f'Best gamma: {best[1]}')
print(f'Accuracy: {score}')


svm = SVC(C=best[0], gamma=best[1], kernel='rbf')

# split the dataset into 70% training set and 30% test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fit the SVM model with the training data
svm.fit(X_train, y_train)

# Predict the test data
y_pred = svm.predict(X_test)

from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score

# Calculate precision
precision = precision_score(y_test, y_pred)
print("\nPrecision: {:.2f}%".format(precision*100))

# Calculate f1-score
f1 = f1_score(y_test, y_pred)
print("F1-Score: {:.2f}%".format(f1*100))

# Calculate recall
recall = recall_score(y_test, y_pred)
print("Recall: {:.2f}%".format(recall*100))

# Calculate accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(acc*100))

# line plot of best scores
pyplot.plot(scores, '.-')
pyplot.xlabel('Improvement Number')
pyplot.ylabel('Evaluation f(x)')
plt.title('Liver')
pyplot.show()
