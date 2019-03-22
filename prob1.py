#!/usr/bin/python

#############################################################
# CS5783: Machine Learning #
# Assignment 3: Linear Regressions #
# Problem 1: Linear regressions: Polynomial basis functions #
# Sk. Mashfiqur Rahman #
#############################################################

import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt("crash.txt")
training = data[1::2]
test = data[0:-1:2]
training_x = np.array(training[:, 0]).reshape(len(training), 1)
training_t = np.array(training[:, 1]).reshape(len(training), 1)
test_x = np.array(test[:, 0]).reshape(len(training), 1)
test_t = np.array(test[:, 1]).reshape(len(training), 1)
Erms_training = np.zeros(20)
Erms_test = np.zeros(20)
Erms_reference_training = 1000.
best_w_training = 0
best_L_training = 0
Erms_reference_test = 1000.
best_w_test = 0
best_L_test = 0


for L in range(1,21):
    phi = training_x**range(L)
    w = np.linalg.solve(phi.T.dot(phi), phi.T.dot(training_t))
    E_training = 0.5 * np.square(np.linalg.norm(training_t - phi.dot(w)))
    Erms_training[L-1] = np.sqrt(2. * E_training / len(training))
    if Erms_training[L-1] < Erms_reference_training:
        Erms_reference_training = Erms_training[L-1]
        best_L_training = L
        best_w_training = w

    phi = test_x**range(L)
    E_test = 0.5 * np.square(np.linalg.norm(test_t - phi.dot(w)))
    Erms_test[L-1] = np.sqrt(2. * E_test / len(training))
    if Erms_test[L-1] < Erms_reference_test:
        Erms_reference_test = Erms_test[L-1]
        best_L_test = L
        best_w_test = w

print('Maximum likelihood RMS error between the actual data and the models prediction (for Training sets): \n')
print(Erms_training)
print('Maximum likelihood RMS error between the actual data and the models prediction (for Test sets): \n')
print(Erms_test)

plt.figure(figsize=(16,12))
plt.plot(Erms_training, '-o', markerfacecolor='none', color='b', label='Training')
plt.plot(Erms_test, '-o', markerfacecolor='none', color='r', label='Test')
plt.suptitle('Maximum likelihood RMS error between the actual data and the models prediction', fontsize=24)
plt.legend(fontsize=22)
plt.xlabel("M ", fontsize = 22)
plt.ylabel("Erms", fontsize = 22)
plt.show()

x = np.linspace(start=np.min(training_x), stop= np.max(training_x), num=100).reshape(100, 1)
phi = x**range(best_L_training)
y = phi.dot(best_w_training)
print('Lowest RMS L for training data:', best_L_training)

plt.figure(figsize=(16,12))
plt.plot(training_x, training_t, color='b', label='Training data')
plt.plot(x, y, color='r', label='Lowest RMS model output')
plt.suptitle('Best fit on the training set', fontsize=24)
plt.legend(fontsize=22)
plt.xlabel("time ", fontsize = 22)
plt.ylabel("acceleration", fontsize = 22)
plt.show()

x = np.linspace(start=np.min(test_x), stop=np.max(test_x), num=100).reshape(100, 1)
phi = x**range(best_L_test)
y = phi.dot(best_w_test)
print('Lowest RMS L for test data:', best_L_test)

plt.figure(figsize=(16,12))
plt.plot(test_x, test_t, color='b', label='Test data')
plt.plot(x, y, color='r', label='Lowest RMS model output')
plt.suptitle('Best fit on the test set', fontsize=24)
plt.legend(fontsize=22)
plt.xlabel("time ", fontsize = 22)
plt.ylabel("acceleration", fontsize = 22)
plt.show()
