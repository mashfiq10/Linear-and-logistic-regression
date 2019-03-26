#!/usr/bin/python

#############################################################
# Logistic regression #
# Sk. Mashfiqur Rahman #
# collect data from http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data #
#############################################################

import numpy as np
from scipy.optimize import minimize


def flower_to_float(s):
    d = {b'Iris-setosa': 0., b'Iris-versicolor': 1., b'Iris-virginica': 2.}
    return d[s]


def phi(_x, _i):
    phi1 = np.ones(5)
    phi1[1:5] = _x[_i,1:5]
    return phi1


def f(_w, training_x, training_y, prior):
    N = training_x.shape[0]
    K = 3
    log_likelihood= 0.
    for n in range(N):
        ll2 = 0.
        ll3 = 0.
        for k in range(K):
            ll2 += training_y[n, k] * _w[5*k:(5*k)+5].T.dot(phi(training_x,n))
        for l in range(K):
            ll3 += np.exp(_w[5*l:(5*l)+5].T.dot(phi(training_x,n)))
        log_likelihood += ll2
        log_likelihood -= np.log(ll3)
        #  forced normalize the overflow encountered in exp
        if log_likelihood == -np.inf:
            log_likelihood = 0.
    return prior - log_likelihood


irises = np.loadtxt('iris.txt', delimiter=',', converters={4:flower_to_float})
M = irises.shape[0]
K = 3
x = np.array(irises[:, :4])
x = np.concatenate((np.ones(shape=(M, 1)), x), axis=1)
y = np.zeros(shape=(M, K))
for i in range(M):
    y[i, int(irises[i,4])] = 1


r = np.arange(M)
np.random.shuffle(r)
x = np.array(x[r.reshape(-1)])
y = np.array(y[r.reshape(-1)])

training_x = x[:int(M/2)]
training_y = y[:int(M/2)]
test_x = x[int(M/2):]
test_y = y[int(M/2):]

alpha = 0.0031257
w_init = np.ones(15)
prior = (alpha/2.) * w_init.T.dot(w_init)

w_hat = minimize(f, w_init, args=(training_x, training_y, prior)).x

b = []
c = []
for i in range(test_x.shape[0]):
    z = np.zeros(K)
    for k in range(K):
        z[k] = np.exp(w_hat[5*k:(5*k)+5].T.dot(phi(test_x,i)))
        #  forced normalize the overflow encountered in exp
        if z[k] == np.inf:
            z[k] = 1000.
    s = z / np.sum(z)
    q = np.amax(s)
    b.append((s.tolist()).index(q))
for i in range(test_y.shape[0]):
    s = test_y[i,:]
    q = np.amax(s)
    c.append((s.tolist()).index(q))
k = np.equal(b,c)
print("overall classification accuracy = ", float(np.sum(k) / len(test_y))*100., '%')
