#!/usr/bin/env python3

import numpy as np
import multiprocessing as mp

def get_sorted_xy(array_X, list_y):
    # sort X and Y
    list_xy = []
    for ii in range(len(list_y)):
        list_xy.append((array_X[ii], list_y[ii]))
    list_xy = sorted(list_xy, key=lambda xy: xy[0])
    return list_xy

def get_data(file_name):
    with open(file_name, 'r') as fr:
        list_lines = fr.readlines()
    # get XY
    DATA_SIZE = len(list_lines)
    DIMENSION = len((list_lines[0].strip()).split())-1
    matrix_X = np.zeros((DIMENSION, DATA_SIZE))
    list_y = []
    for ii in range(DATA_SIZE):
        list_xy = (list_lines[ii].strip()).split()
        list_y.append(float(list_xy[-1]))
        for jj in range(DIMENSION):
            matrix_X[jj][ii] = float(list_xy[jj])
    # sort
    matrix_xy = []
    for ii in range(DIMENSION):
        list_xy = get_sorted_xy(matrix_X[ii], list_y)
        matrix_xy.append(list_xy)
    return matrix_xy, DIMENSION, DATA_SIZE

def decision(list_xy, list_d):
    # find the best dichotomy and Ein
    index_d = -1
    Ein = 10000
    for ii in range(len(list_d)):
        num_e = 0
        for jj in range(len(list_xy)):
            if list_xy[jj][1] != list_d[ii][jj]:
                num_e += 1
        if Ein > num_e:
            Ein = num_e
            index_d = ii
    return index_d, Ein

def get_eout(s, theta, list_xy):
    DATA_SIZE = len(list_xy)
    # find the threshold
    left = 0
    right = DATA_SIZE
    while right-left > 1:
        middle = int((left+right)/2)
        if list_xy[middle][0] < theta:
            left = middle
        elif list_xy[middle][0] > theta:
            right = middle
        else:
            left = right = middle
    list_d = [-s]*left
    list_d.extend([s]*(DATA_SIZE-left))
    err = 0
    for jj in range(DATA_SIZE):
        if list_d[jj] != list_xy[jj][1]:
            err += 1
    return err

def main():
    # load the D_train
    matrix_xy, DIMENSION, DATA_SIZE = get_data("hw2_train.dat")
    # get_dichotomies
    list_dichotomies = []
    for ii in range(DATA_SIZE): # positive ray
        dichotomy = [-1.0]*ii
        dichotomy.extend([1.0]*(DATA_SIZE-ii))
        list_dichotomies.append(dichotomy)
    for ii in range(DATA_SIZE): # negative ray
        dichotomy = [1.0]*ii
        dichotomy.extend([-1.0]*(DATA_SIZE-ii))
        list_dichotomies.append(dichotomy)
    # find the best of the best's
    pool = mp.Pool(processes=mp.cpu_count())
    results = [pool.apply_async(decision, args=(matrix_xy[ii], list_dichotomies)) for ii in range(DIMENSION)]
    output = [p.get() for p in results]
    id_in = -1
    Ein = 10000
    dim_in = -1
    for ii in range(DIMENSION):
        if Ein > output[ii][1]:
            id_in = output[ii][0]
            Ein = output[ii][1]
            dim_in = ii
    Ein /= DATA_SIZE
    print("Ein = {}".format(Ein))
    # find s and theta
    s = 0
    theta = 0
    if id_in <= DATA_SIZE:
        s = 1
        if id_in == 0: # all 1's
            theta = (-11 + matrix_xy[dim_in][id_in][0])/2
        elif id_in == 20: # all (-1)'s
            theta = (11 + matrix_xy[dim_in][id_in][0])/2
        else:
            theta = (matrix_xy[dim_in][id_in-1][0]+matrix_xy[dim_in][id_in][0])/2
    else:
        s = -1
        theta = (matrix_xy[dim_in][id_in-DATA_SIZE-1][0]+matrix_xy[dim_in][id_in-DATA_SIZE][0])/2
    # calculate Eout
    matrix_xy, DIMENSION, DATA_SIZE = get_data("hw2_test.dat")
    results = [pool.apply_async(get_eout, args=(s, theta, matrix_xy[ii])) for ii in range(DIMENSION)]
    output = [p.get() for p in results]
    Eout = min(output)/DATA_SIZE
    print("Eout = {}".format(Eout))
    return    
    
if __name__ == "__main__":
    main()
