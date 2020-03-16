# https://bit.ly/379wq2t

import numpy as np
import math
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
#import multiprocessing as mp

def load_x(list_line):
    list_x = []
    for ii in range(len(list_line)):
        data = list_line[ii].strip().split()
        list_x.append(list(map(float, data[:-1]))) # x
    return np.array(list_x)

def main():
    # load data
    file_data = "hw4_nolabel_train.dat"
    x_train = None
    with open(file_data, 'r') as fr:
        list_line = fr.readlines()
        x_train = load_x(list_line)

    num_exp = 500
    list_k = [2, 4, 6, 8, 10]
    list_ein = []
    for kk in list_k:
        for nn in range(num_exp):
            kmeans = KMeans(n_clusters=kk, init="random")
            kmeans.fit(x_train)
            distance = kmeans.transform(x_train)
            labels = kmeans.labels_
            print("test")

    
    return

if __name__ == "__main__":
    main()

