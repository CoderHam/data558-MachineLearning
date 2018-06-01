import numpy as np
from numpy import linalg as LA
import copy
import scipy.linalg
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

class LinearSVM():
    """
        A linear support vector machine with the huberized/smoothed hinge loss
        Note: It is a binary classifier. A multiclass one will be built with it
    """

    def obj(self,beta,lam,x,y,h):
        """
            Objective function:
            lambda * l2_norm(beta)^2 + 1/n * sum(llh(yi,xi.T*B))
            where n is the number of samples, beta is the vector of weights,
            i ranges from 1 to n and llh(y_i,t_i) is defined as :-
                0                            if (y_i*t_i) > 1-h
                ((1+h-(y_i*t_i))^2)/4h       if abs(1-(y_i*t_i)) <= h
                1-(y_i*t_i)                  if (y_i*t_i) < 1-h
            where h can take any value with the default set to 0.5
            And t_i = x_i.T*beta
        """

        n,d = x.shape
        yt = y*np.dot(x, beta)
        llh = (1+h-yt)**2/(4*h)*(np.abs(1-yt) <= h) + (1-yt)*(yt < (1-h))
        return 1/n*np.sum(llh) + lam*(beta.dot(beta))


    def compute_grad(self,beta,lam,x,y,h):
        """
            Gradient function:
            2*lambda*beta + 1/n * sum(llh_grad(yi,xi.T*B))
            where n is the number of samples, beta is the vector of weights,
            i ranges from 1 to n and llh_grad(y_i,t_i) is defined as :-
                0                                    if (y_i*t_i) > 1-h
                -(1+h-(y_i*t_i))^2)/2h*y-i*x_i       if abs(1-(y_i*t_i)) <= h
                -y-i*x_i                             if (y_i*t_i) < 1-h
            where h can take any value with the default set to 0.5
            And t_i = x_i.T*beta
        """

        n,d = x.shape
        yt = y*np.dot(x, beta)
        ll_grad = -(1+h-yt)/(2*h)*y*(np.abs(1-yt) <= h)-y*(yt < (1-h))
        return 1/n*np.sum(ll_grad[:, np.newaxis]*x,axis=0) + 2*lam*beta

    def backtracking_ls(self,theta_,t,lam,x,y,alpha_=0.5,beta_=0.5,\
                        h=0.5,max_iter=100):
        """
            Backtracking is used to reduce the step size as training progresses
            Returns: new step size
        """

        grad = self.compute_grad(beta=theta_,lam=lam,x=x,y=y,h=h)
        norm = LA.norm(grad)
        found_t = False
        i = 0
        while (found_t is False and i < max_iter):
            if self.obj(theta_-t*grad,lam,x,y,h) < \
                (self.obj(theta_,lam,x,y,h) - alpha_*t*(norm**2)):
                found_t = True
            elif i == max_iter - 1:
                tmp_theta = theta_
                break
            else:
                t *= beta_
                i += 1
        return t

    def fit(self,X,Y,lam=1,h=0.5,eps=1e-5,max_iter=1000):
        """
            The fast gradient algorithm is used to train the model.
            default: lambda = 1, maximum iterations = 1000
            Returns: final weights, intermediate weights, intermediate objective
        """

        n,d = X.shape
        # if lambda is None, set to optimal lambda using cross validation
        if lam is None:
            lam = self.cross_validate(X,Y)
            print("Setting optimal value for lambda as:",lam)
        # initial t is set to lipschitz constant
        t = 1/scipy.linalg.eigh(2/n*np.dot(X.T,X) + 2*lam,\
                             eigvals=(d-1,d-1), eigvals_only=True)[0]
        # initial weights are set to zero
        beta = np.zeros(d)
        theta = np.zeros(d)
        grad_theta = self.compute_grad(theta,lam,X,Y,h)
        beta_vals = copy.deepcopy(beta)
        obj_vals = self.obj(beta=beta,lam=lam,x=X,y=Y,h=h)
        i = 1
        while i < max_iter and LA.norm(grad_theta) > eps:
            t = self.backtracking_ls(theta_=theta,lam=lam,t=t,x=X,y=Y)
            beta = theta - t*self.compute_grad(beta=theta,lam=lam,x=X,y=Y,h=h)
            beta_vals = np.vstack((beta_vals,beta))
            theta = beta_vals[i] + i/(i+3)*(beta-beta_vals[i-1])
            obj_vals = np.append(obj_vals,self.obj(beta,lam,X,Y,h))
            grad_theta = self.compute_grad(theta,lam,X,Y,h)
            i+=1
        return beta, beta_vals, obj_vals

    def plot_objective(self,obj_vals):

        """
            Plots the change in objective values with each iteration
        """
        fig = plt.gcf()
        fig.set_size_inches(12,9)
        plt.plot(np.arange(len(obj_vals)),obj_vals,label="huberized hinge loss")
        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.legend(loc='upper right')
        plt.title('Objective function vs Iterations')
        plt.show()

    def calc_error(self, beta, x, y):
        """
            Calculates the misclassification error
        """
        y_pred = 1 / (1 + np.exp(-x.dot(beta))) > 0.5
        y_pred = y_pred * 2 - 1
        return np.mean(y_pred != y)

    def plot_training_error(self,betas,x,y,x2,y2):
        """
            Plots misclassification error for train and test with each iteraion
        """

        length = np.size(betas, 0)
        error_train = np.zeros(length)
        error_val = np.zeros(length)
        for i in range(length):
            error_train[i] = self.calc_error(betas[i], x, y)
            error_val[i] = self.calc_error(betas[i], x2, y2)

        fig = plt.figure(figsize=(12, 9))
        plt.plot(range(length), error_train, c='blue', label='train')
        plt.plot(range(length), error_val, c='red', label='test/val')
        plt.xlabel('Number of Iteration')
        plt.ylabel('Misclassification error')
        plt.legend(loc='upper right')
        plt.title('Train and Test error vs Iterations')
        plt.show()
        print("Final train error:",error_train[i])
        print("Final test error:",error_val[i])

    def predict(self,beta,x):
        """
            Predicts labels given weights and features
        """
        y_pred = 1 / (1 + np.exp(-x.dot(beta))) > 0.5
        y_pred = y_pred * 2 - 1
        return y_pred

    def cross_validate(self,X,Y,k=3):
        """
            Perform k-fold cross validation to choose best value of lambda.
            Use mean of best lambda for all k folds as optimal lambda
        """
        kf = KFold(n_splits=k)
        lambda_vals = [2**k for k in range(-10, 5)]
        lambda_optimal = []

        for train_i, test_i in kf.split(Y):
            Xtr, Xts = X[train_i], X[test_i]
            ytr, yts = Y[train_i], Y[test_i]
            n, d = Xtr.shape
            x_init = np.zeros(d)
            train_errors = np.zeros(15)
            test_errors = np.zeros(15)
            for i in range(0,15):
                li=lambda_vals[i]
                b, betas, objs = self.fit(X,Y,lam=li,max_iter=100)
                train_errors[i] = self.calc_error(b, Xtr, ytr)
                test_errors[i] = self.calc_error(b, Xts, yts)

            lambda_optimal.append(lambda_vals[np.argmin(test_errors)])
        return np.mean(lambda_optimal)
