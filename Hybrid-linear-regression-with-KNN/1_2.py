# -*- coding: utf-8 -*-
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from scipy.misc import logsumexp

np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506, 1)), x), axis=1)  # add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

# helper function
def l2(A, B):
    '''
    Input: A is a Nxd matrix
           B is a M*d matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
    B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
    dist = A_norm + B_norm - 2 * A.dot(B.transpose())
    return dist


# helper function
def run_on_fold(x_test, y_test, x_train, y_train, taus):
    '''
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses one for each tau value
    '''
    N_test = x_test.shape[0]
    losses = np.zeros(taus.shape)
    for j,tau in enumerate(taus):
        predictions =  np.array([LRLS(x_test[i,:].reshape(d,1),x_train,y_train, tau) \
                        for i in range(N_test)])
        losses[j] = ((predictions.flatten()-y_test.flatten())**2).mean()
    return losses


# to implement
def LRLS(test_datum, x_train, y_train, tau, lam=1e-5):
    '''
    Input: test_datum is a d * 1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    a = np.zeros(shape = (x_train.shape[0],1))
    A = np.zeros(shape = (a.shape[0],a.shape[0]))
    test_datum = test_datum.reshape((1,test_datum.shape[0]))

    dist_t = l2(x_train,test_datum)
    dist = dist_t*(-1/(2*tau**2))
    dist_c = dist - dist.max()
    a=np.exp(dist_c)/np.sum(np.exp(dist_c))
    for i in range(a.shape[0]):
        A[i,i] = a[i,0]

    IdentitySize = x_train.shape[1]
    IdentityMatrix = np.zeros((IdentitySize, IdentitySize))
    np.fill_diagonal(IdentityMatrix, 1)
    XtAX_lam = np.dot(np.dot(x_train.T,A),x_train) + lam * IdentityMatrix
    XtAY = np.dot(np.dot(x_train.T,A),y_train)
    w = np.linalg.solve(XtAX_lam, XtAY)
    w = w.reshape((w.shape[0],1))
    y_out = np.dot(test_datum[0], w)
    return y_out

def run_k_fold(x, y, taus, k):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    '''
    idx_train = np.zeros((N - N//k,1))
    idx_test = np.zeros((N//k,1))
    losses = np.zeros(shape = (5,taus.shape[0]))
    n = N // k
    for i in range(k):
        idx_test = idx[(i * n):((i + 1) * n)]
        idx_train = np.concatenate((idx[0:i * n], idx[((i + 1) * n):(-1)]))
        x_test = x[idx_test,:]
        x_train = x[idx_train,:]
        y_test = y[idx_test]
        y_train = y[idx_train]
        losses[i,:] = run_on_fold(x_test, y_test, x_train, y_train, taus)
    return losses


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value.
    taus = np.logspace(1.0, 3, 200)
    loss = run_k_fold(x, y, taus, k=5)
    print(loss.shape)
    losses = np.mean(loss,axis = 0)
    plt.figure(0)
    plt.plot(taus,losses,'.')
    plt.xlabel('τ')
    plt.ylabel('average loss')
    print("min loss = {}".format(losses.min()))
    plt.show()
