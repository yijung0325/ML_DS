# https://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_twoclass.html

import numpy as np
import multiprocessing as mp
import math
import matplotlib.pyplot as plt

def load_xy(list_line):
    matrix_xy = []
    for ii in range(len(list_line)):
        data = list_line[ii].strip().split()
        list_xy = list(map(float, data[:-1]))
        list_xy.append(int(data[-1]))
        matrix_xy.append(list_xy)
    return np.array(matrix_xy)

def main():
    # load data
    file_data = "hw2_adaboost_train.dat"
    xy_train = None
    with open(file_data, 'r') as fr:
        list_line = fr.readlines()
        xy_train = load_xy(list_line)
    
    
    return

if __name__ == "__main__":
    main()

