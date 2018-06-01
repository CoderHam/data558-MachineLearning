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
# Loading the spam data
spam = pd.read_csv("data/spam/spam.csv",header=None,sep=" ")
train_or_test = np.loadtxt("data/spam/spam.traintest.txt")
print("Total number of records:",len(spam))

# All but last column are the features that we will use to predict
X = spam.iloc[:,:-1]
# Last column is the label - 1 means spam, 0 means not spam
Y = spam.iloc[:,-1]
Y = np.array([-1 if yi==0 else 1 for yi in Y])
# split the data into train and test using the train:test ratio of 0.75 and 0.25 (shuffle the data)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0,shuffle=True)
train_id, test_id = train_or_test==0, train_or_test==1
X_train, X_test, Y_train, Y_test = X[train_id], X[test_id], Y[train_id], Y[test_id]

# Normalize/Standardize the data using sklearn
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
print("Saving predictions in spam_test_predict.npy...")
np.save("spam_test_predict.npy",ypred_rectified)
print("Accuracy on test set:",np.mean(ypred==Y_test))
