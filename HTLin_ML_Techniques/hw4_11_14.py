# https://bit.ly/379wq2t

import numpy as np
import math
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
#import multiprocessing as mp

def yhat_uniform(x_data, x_train, y_train, gamma):
    result = 0
    for ii in range(x_data.shape[0]):
        distance2 = (np.linalg.norm(x_data - x_train[ii]))**2
        result += y_train[ii]*math.exp(-gamma*distance2)
    if result >= 0:
        return 1
    else:
        return -1

def get_err(knn, x_data, y_data):
    num_err = 0
    for ii in range(x_data.shape[0]):
        y_hat = knn.predict([x_data[ii]])
        if y_hat != y_data[ii]:
            num_err += 1
    return num_err/y_data.size

def load_xy(list_line):
    list_x = []
    list_y = []
    for ii in range(len(list_line)):
        data = list_line[ii].strip().split()
        list_x.append(list(map(float, data[:-1]))) # x
        list_y.append(int(data[-1]))
    return np.array(list_x), np.array(list_y)

def main():
    # load data
    file_data = "hw4_train.dat"
    x_train = None
    y_train = None
    with open(file_data, 'r') as fr:
        list_line = fr.readlines()
        x_train, y_train = load_xy(list_line)
    file_data = "hw4_test.dat"
    x_test = None
    y_test = None
    with open(file_data, 'r') as fr:
        list_line = fr.readlines()
        x_test, y_test = load_xy(list_line)

    # Q11, Q12
    # list_k = [1, 3, 5, 7, 9]
    # list_ein = []
    # list_eout = []
    # for ii in list_k:
    #     knn = KNeighborsClassifier(n_neighbors=ii)
    #     knn.fit(x_train, y_train)
    #     list_ein.append(get_err(knn, x_train, y_train))
    #     list_eout.append(get_err(knn, x_test, y_test))
    # plt.xlabel("k")
    # plt.ylabel("Error")
    # plt.plot(list_k, list_ein, label="Ein")
    # plt.plot(list_k, list_eout, label="Eout")
    # plt.legend()
    # plt.savefig("ein_eout.png")

    # Q13, Q14
    list_gamma = [0.001, 0.01, 0.1, 1, 10, 100]
    list_ein = []
    list_eout = []
    for gamma in list_gamma:
        num_err = 0
        for ii in range(x_train.shape[0]):
            yhat = yhat_uniform(x_train[ii], x_train, y_train, gamma)
            if yhat != y_train[ii]:
                num_err += 1
        list_ein.append(num_err/x_train.shape[0])
        num_err = 0
        for ii in range(x_test.shape[0]):
            yhat = yhat_uniform(x_test[ii], x_train, y_train, gamma)
            if yhat != y_test[ii]:
                num_err += 1
        list_eout.append(num_err/x_test.shape[0])
    plt.xlabel("Gamma")
    plt.ylabel("Error")
    plt.xscale("log")
    plt.plot(list_gamma, list_ein, label="Ein")
    plt.plot(list_gamma, list_eout, label="Eout")
    plt.legend()
    plt.savefig("uni_ein_eout.png")
        

    
    return

if __name__ == "__main__":
    main()

