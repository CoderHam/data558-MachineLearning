import numpy as np
from numpy import linalg as LA
import copy
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

from itertools import combinations
from collections import Counter

from models import linear_svm
np.random.seed(4444)
# Loading the spam data
vowel_train = pd.read_csv("data/vowel/vowel_train.csv",sep=",")
vowel_test = pd.read_csv("data/vowel/vowel_test.csv",sep=",")
# All but first column are the features that we will use to predict
X_train = vowel_train.iloc[:,1:]
X_test = vowel_test.iloc[:,1:]
# First column is the label - ranges from 1 to 11
Y_train = vowel_train.iloc[:,0]
Y_test = vowel_test.iloc[:,0]
print("Total number of records:",len(vowel_train)+len(vowel_test))

# Normalize/Standardize the data using sklearn
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pairs = list(combinations(list(set(Y_train)), 2))

# training the model using my custom SVM
print("Training one-vs-one classifiers for the",len(pairs),"(",len(set(Y_train)),"C2) pairs...")

LSVM = linear_svm.LinearSVM()
weights = []
weight_vals = []
objective_vals = []
for pi in pairs:
    print("\nClassifier for pair",pi,":")
    c1_index = np.where(Y_train==pi[0])[0]
    c2_index = np.where(Y_train==pi[1])[0]
    v1_index = np.where(Y_test==pi[0])[0]
    v2_index = np.where(Y_test==pi[1])[0]

    X_train_c1 = X_train[c1_index]
    X_train_c2 = X_train[c2_index]
    X_test_c1 = X_test[v1_index]
    X_test_c2 = X_test[v2_index]
    X_train_c12 = np.concatenate((X_train_c1,X_train_c2),axis=0)
    X_test_c12 = np.concatenate((X_test_c1,X_test_c2),axis=0)

    Y_train_c12 = np.array([-1]*len(c1_index)+[1]*len(c2_index))
    Y_test_c12 = np.array([-1]*len(v1_index)+[1]*len(v2_index))

    X_train_c12, Y_train_c12 = shuffle(X_train_c12, Y_train_c12, random_state=0)
    w, w_vals, obj_vals = LSVM.fit(X=X_train_c12,Y=Y_train_c12,lam=None)
    weights.append(w)
    weight_vals.append(w_vals)
    objective_vals.append(obj_vals)
    print("Final train error:",LSVM.calc_error(w,X_train_c12,Y_train_c12))
    print("Final test error:",LSVM.calc_error(w,X_test_c12,Y_test_c12))

    # Uncomment to plot objective function after each iteraion
    # LSVM.plot_objective(obj_vals)
    # Uncomment to plot train and test error after each iteraion
    # LSVM.plot_training_error(w_vals,X_train,Y_train,X_test,Y_test)

# Uncomment to save learned weights
# np.save("vowel_ovo_weights.npy",weights)
# weights = np.load("vowel_ovo_weights.npy")

# Use the binary classifiers to build a multiclass classifier (One-vs-One)
def multi_predict(X_features):
    ypred = np.zeros((len(weights),len(X_features)))
    for wi in range(len(weights)):
        ypred_temp = LSVM.predict(weights[wi],X_features)
        ypred[wi] = [pairs[wi][0] if yi==-1 else pairs[wi][1] for yi in ypred_temp]
    ypred = ypred.T
    ypred_final = [Counter(fp).most_common(1)[0][0] for fp in ypred]
    return ypred_final
# Saving the predictions in a numpy array
ypred_final = multi_predict(X_test)

print("Saving predictions in vowel_test_predict.npy...")
np.save("vowel_test_predict.npy",ypred_final)
ypred_train = multi_predict(X_train)
print("Accuracy on train set:",np.mean(ypred_train==Y_train))
print("Accuracy on test set:",np.mean(ypred_final==Y_test))
