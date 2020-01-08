# https://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_twoclass.html

import numpy as np
import multiprocessing as mp
import math
import matplotlib.pyplot as plt

class DECISION_STUMP:
    def __init__(self, ss, ii, nn, array_x):
        self.ss = ss
        self.ii = ii
        if nn == 0:
            self.theta = array_x[0]/2
        else:
            self.theta = (array_x[nn-1]+array_x[nn])/2
        self.array_yhat = np.append([-ss]*nn, [ss]*(array_x.size-nn))
        self.array_yhat = self.array_yhat.astype(int)

    def get_ein(self, array_y, array_id, array_u):
        array_eid = array_id[array_y != self.array_yhat] # the indices of errors
        Ein = np.sum(array_u[array_eid])/np.sum(array_u)
        return [Ein, array_eid]

    def classify(self, xx):
        if xx >= self.theta:
            return self.ss
        else:
            return -self.ss

def load_xy(list_line):
    x = [] # with feature(s)
    y = [] # with class and index
    index = [] # with class and index
    for ii in range(len(list_line)):
        data = list_line[ii].strip().split()
        x.append(list(map(float, data[:-1])))
        y.append(int(data[-1]))
        index.append(ii)
    # sort
    matrix_xy = []
    for ii in range(len(x[0])): # the number of features
        list_xy = []
        for jj in range(len(y)): # the size of data
            list_xy.append([x[jj][ii], y[jj], index[jj]])
        list_xy = sorted(list_xy, key=lambda xy: xy[0])
        matrix_xy.append(list_xy)
    matrix_xy = np.array(matrix_xy)
    tm_xy = np.zeros((len(x[0]), 3, len(y))) # (features, x+y+id, size)
    for ii in range(len(x[0])):
        tm_xy[ii] = np.transpose(matrix_xy[ii])
    return tm_xy

def main():
    # load data
    file_data = "hw2_adaboost_train.dat"
    xy_train = None
    with open(file_data, 'r') as fr:
        list_line = fr.readlines()
        xy_train = load_xy(list_line)
    file_data = "hw2_adaboost_test.dat"
    xy_test = None
    with open(file_data, 'r') as fr:
        list_line = fr.readlines()
        xy_test = load_xy(list_line)
    num_iter = 300

    # initiate
    list_ds = [] # all possible decision stumps
    for ii in range(xy_train.shape[0]): # the feature index
        for ss in range(-1, 2, 2): # the sign of a decision stump: -1 or 1
            list_ds.append(DECISION_STUMP(ss, ii, 0, xy_train[ii][0]))
            for nn in range(1, xy_train.shape[2]): # the index of an interval
                list_ds.append(DECISION_STUMP(ss, ii, nn, xy_train[ii][0]))
    array_u = np.full(xy_train.shape[2], 1/xy_train.shape[2])
    array_decision_Gt_in = np.zeros(xy_train.shape[2]) # for each data points, record their decision during each iteration
    array_decision_Gt_out = np.zeros(xy_test.shape[2]) # for each data points, record their decision during each iteration
    # run
    pool = mp.Pool(processes=mp.cpu_count())
    list_ein_gt = [] # Q13
    list_ein_Gt = [] # Q14
    list_ut = [1] # Q15
    list_eout_Gt = [] # Q16
    list_alpha = []
    list_gt_index = []
    for tt in range(num_iter):
        results = []
        for gg in range(len(list_ds)):
            if gg < len(list_ds)/2:
                ii = 0
            else:
                ii = 1
            results.append(pool.apply_async(list_ds[gg].get_ein, args=(xy_train[ii][1].astype(int), xy_train[ii][2].astype(int), array_u)))
        ein_eid = np.transpose([rr.get() for rr in results])
        # update array_u
        Ein_gt = np.min(ein_eid[0])
        gt_index = np.argmin(ein_eid[0])
        factor = ((1-Ein_gt)/Ein_gt)**0.5
        array_eid = np.sort(ein_eid[1][gt_index])
        pp = 0
        for ii in range(array_u.size):
            if pp<array_eid.size and ii==array_eid[pp]: # error
                array_u[ii] *= factor
                pp += 1
            else:
                array_u[ii] /= factor
        # dump results
        list_ein_gt.append(Ein_gt)
        list_ut.append(np.sum(array_u))
        list_alpha.append(math.log(factor))
        list_gt_index.append(gt_index)
        # calculate Gt_in
        results = []
        gt = list_ds[gt_index]
        for xx in range(xy_train.shape[2]):
            results.append(pool.apply_async(gt.classify, args=(xy_train[gt.ii][0][xx], )))
        array_decision_gt = np.array([rr.get() for rr in results])
        for rr in range(array_decision_gt.size):
            index_original = xy_train[gt.ii][2][rr].astype(int)
            array_decision_Gt_in[index_original] += array_decision_gt[rr]*list_alpha[-1]
        # calculate Ein(Gt)
        array_yid = xy_train[gt.ii][1:]
        array_y = array_yid[0][np.argsort(array_yid[1])]
        array_result = array_decision_Gt_in*array_y
        Ein_Gt = (array_result[array_result < 0]).size/xy_train.shape[2]
        list_ein_Gt.append(Ein_Gt)
        # calculate Gt_out
        results = []
        for xx in range(xy_test.shape[2]):
            results.append(pool.apply_async(gt.classify, args=(xy_test[gt.ii][0][xx], )))
        array_decision_gt = np.array([rr.get() for rr in results])
        for rr in range(array_decision_gt.size):
            index_original = xy_test[gt.ii][2][rr].astype(int)
            array_decision_Gt_out[index_original] += array_decision_gt[rr]*list_alpha[-1]
        # calculate Eout(Gt)
        array_yid = xy_test[gt.ii][1:]
        array_y = array_yid[0][np.argsort(array_yid[1])]
        array_result = array_decision_Gt_out*array_y
        Eout_Gt = (array_result[array_result < 0]).size/xy_test.shape[2]
        list_eout_Gt.append(Eout_Gt)
        print("{} {} {} {} {}".format(tt, round(Ein_gt, 5), Ein_Gt, round(list_ut[-2], 5), Eout_Gt))
    pool.close()

    # Q13
    # print("Ein(gT) = {}".format(list_ein_gt[-1]))
    # plt.xlabel("t")
    # plt.ylabel("Ein(gt)")
    # plt.plot(list(range(num_iter)), list_ein_gt)
    # plt.savefig("Ein_gt.png")

    # Q14
    # print("Ein(GT) = {}".format(list_ein_Gt[-1]))
    # plt.xlabel("t")
    # plt.ylabel("Ein(Gt)")
    # plt.plot(list(range(num_iter)), list_ein_Gt)
    # plt.savefig("Ein_Gt.png")

    # Q15
    # print("UT = {}".format(list_ut[-2]))
    # plt.xlabel("t")
    # plt.ylabel("Ut")
    # plt.plot(list(range(num_iter)), list_ut[:-1])
    # plt.savefig("Ut.png")

    # Q16
    print("Eout(GT) = {}".format(list_eout_Gt[-1]))
    plt.xlabel("t")
    plt.ylabel("Eout(Gt)")
    plt.plot(list(range(num_iter)), list_eout_Gt)
    plt.savefig("Eout_Gt.png")
    
    
    return

if __name__ == "__main__":
    main()

