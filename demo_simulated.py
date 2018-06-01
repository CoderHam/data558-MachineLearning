import numpy as np
from numpy import linalg as LA
import copy
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

from models import linear_svm
np.random.seed(4444)

# Generate the simulated data
def generate_data(n=100,d=10,num_class=2):
    """
        Generates simulated data
        default: 100 data points, dimensionality: 10, 2 classes
    """
    series = []
    labels = []
    for ci in range(num_class):
        y = ci
        means = np.random.randn(d)
        sigmas = np.random.randn(d)
        # mean shift using np.random.uniform(60)
        shift_means = means + np.random.uniform(0,1,d)
        # Adding noise to the data
        noise = np.random.normal(0,1,(100,10))
        i = 1
        mini_series = np.random.normal(shift_means,abs(sigmas),(100,10)) + noise
        # print(mini_series.shape)
        series.extend(list(mini_series))
        labels.extend([ci]*n)
    return np.array(series), np.array(labels)

X, Y = generate_data(100,10,2)
Y = np.array([-1 if yi==0 else 1 for yi in Y])
# split the data into train and test using the train:test ratio of 0.75 and 0.25 (shuffle the data)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0,shuffle=True)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Training model...")
LSVM = linear_svm.LinearSVM()
# training the model using my custom SVM
w, w_vals, obj_vals = LSVM.fit(X=X_train,Y=Y_train,lam=None)

# plot objective function after each iteraion
LSVM.plot_objective(obj_vals)
# plot train and test error after each iteraion
LSVM.plot_training_error(w_vals,X_train,Y_train,X_test,Y_test)

ypred = LSVM.predict(w,X_test)
ypred_rectified = [0 if yi==-1 else 1 for yi in ypred]
# Saving the predictions in a numpy array
print("Saving predictions in simulated_test_predict.npy...")
np.save("simulated_test_predict.npy",ypred_rectified)
print("Accuracy on test set:",np.mean(ypred==Y_test))

# Compare to sklearn. Comment out if not needed
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
lsvc = LinearSVC()
parameters = {'C':[2**k for k in range(-10, 5)]}
sklearn_svc = GridSearchCV(lsvc,parameters)
sklearn_svc.fit(X_train,Y_train)
ypred_sk = sklearn_svc.predict(X_test)
print("Accuracy on test set using sklearn:",np.mean(ypred_sk==Y_test))
print("Error between custom and sklearn:",abs(np.mean(ypred_sk==Y_test)-np.mean(ypred==Y_test)))
