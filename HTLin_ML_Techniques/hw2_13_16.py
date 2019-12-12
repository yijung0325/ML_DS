# https://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_twoclass.html

import numpy as np
import multiprocessing as mp
#import math
#import matplotlib.pyplot as plt

class DECISION_STUMP:
    def __init__(self, ss, ii, nn, array_x):
        self.ss = ss
        self.ii = ii
        if nn == 0:
            self.theta = array_x[0]/2
        else:
            self.theta = (array_x[nn-1]+array_x[nn])/2
        self.array_yhat = np.array([-ss]*nn)
        self.array_yhat = np.append(self.array_yhat, [ss]*(array_x.size-nn))

    def get_err(self, array_y, array_u):
        array_e = (array_y != self.array_yhat)
        error = np.sum(array_u[array_e])/np.sum(array_u)
        return [error, array_e]

def load_xy(list_line):
    x = [] # with feature(s)
    y = [] # with class and index
    for ii in range(len(list_line)):
        data = list_line[ii].strip().split()
        x.append(list(map(float, data[:-1])))
        y.append([int(data[-1]), ii])
    # sort
    matrix_xy = []
    for ii in range(len(x[0])):
        list_xy = []
        for jj in range(len(y)):
            list_xy.append([x[jj][ii], y[jj][0], y[jj][1]])
        list_xy = sorted(list_xy, key=lambda xy: xy[0])
        matrix_xy.append(list_xy)
    return np.array(matrix_xy)

def main():
    # load data
    file_data = "hw2_adaboost_train.dat"
    xy_train = None
    with open(file_data, 'r') as fr:
        list_line = fr.readlines()
        xy_train = load_xy(list_line)
    num_iter = 1
    # initiate
    list_gs = []
    for ii in range(xy_train.shape[0]): # the feature index
        for ss in range(-1, 2, 2): # the sign of a decision stump: -1 or 1
            list_gs.append(DECISION_STUMP(ss, ii, 0, xy_train[ii, :, 0]))
            for nn in range(1, xy_train.shape[1]): # the index of an interval
                list_gs.append(DECISION_STUMP(ss, ii, nn, xy_train[ii, :, 0]))
    array_u = np.full(xy_train.shape[1], 1/xy_train.shape[1])
    # Q13
    pool = mp.Pool(processes=mp.cpu_count())
    list_gtein = []
    results = []
    for tt in range(num_iter):
        for gg in range(len(list_gs)):
            if gg < len(list_gs)/2:
                ii = 0
            else:
                ii = 1
            results.append(pool.apply_async(list_gs[gg].get_err, args=(xy_train[ii, :, 1], array_u)))
        ein_ei = np.array([rr.get() for rr in results])
#    print("Ein = {}, gt(s, i, theta) = ({}, {}, {})".format(round(Ein, 5), list_gs[id_gs].ss, list_gs[id_gs].ii, list_gs[id_gs].theta))
#        list_gtein.append(min(list_ein))
#        gt = list_gs[list_gs.index(list_gtein[-1])]
        
    pool.close()

    # plt.xlabel("t")
    # plt.ylabel("Ein")
    # plt.plot(list(range(1, num_iter+1)), list_ein)
    # plt.savefig("t_ein.png")
        
    file_data = "hw2_adaboost_test.dat"
    xy_test = None
    with open(file_data, 'r') as fr:
        list_line = fr.readlines()
        xy_test = load_xy(list_line)
    
    
    return

if __name__ == "__main__":
    main()

