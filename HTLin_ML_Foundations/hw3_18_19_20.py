#!/usr/bin/env python3

import numpy as np
import math

def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

def get_data(file_data):
    with open(file_data, 'r') as fr:
        list_line = fr.readlines()
    list_x = []
    list_y = []
    for line in list_line:
        data = (line.strip()).split()
        list_x.append([])
        for x in data[:-1]:
            list_x[-1].append(float(x))
        list_y.append(float(data[-1]))
    return np.array(list_x), np.array(list_y)

def theta(ss):
    return 1/(1+math.exp(-ss))

def gradient_ein(w, x, y):
    g_ein = np.zeros(x[0].size)
    N = y.size
    for ii in range(N):
        g_ein += theta(-y[ii]*np.dot(w, x[ii]))*(-y[ii]*x[ii])
    return g_ein/N

def get_eout(w_train):
    x_test, y_test = get_data("hw3_test.dat")
    Eout = 0
    N = y_test.size
    for ii in range(N):
        if y_test[ii] != sign(np.dot(w_train, x_test[ii])):
            Eout += 1
    Eout /= N
    return Eout    

def gd_18_19(eta, T, x_train, y_train):
    w_train = np.zeros(x_train[0].size)
    for ii in range(T):
        g_ein = gradient_ein(w_train, x_train, y_train)
#        norm_gein = np.linalg.norm(g_ein)
#        w_train -= (eta/norm_gein)*g_ein
        w_train -= eta*g_ein
    return get_eout(w_train)

def main():
    x_train, y_train = get_data("hw3_train.dat")

    # Q18
    eta = 0.001
    T = 2000
    Eout = gd_18_19(eta, T, x_train, y_train)
    print("Eout_q18 = {}".format(Eout))

    # Q19
    eta = 0.01
    T = 2000
    Eout = gd_18_19(eta, T, x_train, y_train)
    print("Eout_q19 = {}".format(Eout))

    # Q20
    eta = 0.001
    T = 2000
    w_train = np.zeros(x_train[0].size)
    for ii in range(T):
        jj = ii%y_train.size
        w_train += eta*theta(-y_train[jj]*np.dot(w_train, x_train[jj]))*y_train[jj]*x_train[jj]
    Eout = get_eout(w_train)
    print("Eout_q20 = {}".format(Eout))

    return

if __name__ == "__main__":
    main()
