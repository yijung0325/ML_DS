#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random

def get_dichotomies(DATA_SIZE):
    list_dichotomies = []
    # positive ray
    for ii in range(DATA_SIZE):
        dichotomy = [-1]*ii
        dichotomy.extend([1]*(DATA_SIZE-ii))
        list_dichotomies.append(dichotomy)
    # negative ray
    for ii in range(DATA_SIZE):
        dichotomy = [1]*ii
        dichotomy.extend([-1]*(DATA_SIZE-ii))
        list_dichotomies.append(dichotomy)
    return list_dichotomies

def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

def get_data(DATA_SIZE):
    # x
    list_x = []
    for ii in range(DATA_SIZE):
        x = random.uniform(-1, 1)
        while x == 0:
            x = random.uniform(-1, 1)
        list_x.append(x)
    list_x.sort()
    # y
    list_y = []
    for ii in range(DATA_SIZE):
        s = random.random()
        if s < 0.2:
            s = -1
        else:
            s = 1
        list_y.append(s*sign(list_x[ii]))
    return list_x, list_y

def decision(list_x, list_y, list_d):
    # find the dichotomy and Ein
    index_d = -1
    Ein = 10000
    for ii in range(len(list_d)):
        num_e = 0
        for jj in range(len(list_y)):
            if list_y[jj]*list_d[ii][jj] < 0:
                num_e += 1
        if Ein > num_e:
            Ein = num_e
            index_d = ii
    Ein /= len(list_y)
    # Eout
    s = 0
    theta = 0
    if index_d <= len(list_y):
        s = 1
        if index_d == 0: # all 1's
            theta = (-1 + list_x[0])/2
        elif index_d == 20: # all (-1)'s
            theta = (1 + list_x[-1])/2
        else:
            theta = (list_x[index_d-1]+list_x[index_d])/2
    else:
        s = -1
        theta = (list_x[index_d-len(list_y)-1]+list_x[index_d-len(list_y)])/2
    Eout = 0.5+0.3*s*(abs(theta)-1)
    return Ein, Eout

def main():
    DATA_SIZE = 20
    NUM_EXP = 5000
    list_d = get_dichotomies(DATA_SIZE)

    mean_Ein = 0
    mean_Eout = 0
    for ii in range(NUM_EXP):
        list_x, list_y = get_data(DATA_SIZE)
        Ein, Eout = decision(list_x, list_y, list_d)
#        print("[{}]Ein = {}, Eout = {}".format(ii, Ein, Eout))
        mean_Ein += Ein
        mean_Eout += Eout
    print("mean_Ein = {}, mean_Eout = {}".format(mean_Ein/NUM_EXP, mean_Eout/NUM_EXP))

if __name__ == "__main__":
    main()