# https://bit.ly/379wq2t

import numpy as np
import multiprocessing as mp
#import matplotlib.pyplot as plt

class DECISION_STUMP:
    def __init__(self, ss, nn):
        self.nn = nn
        self.ss = ss

    def get_gi(self, array_y): # Gini Impurity
        gi = 0
        if self.nn > 0: # left
            array_yhat_part = np.array([-self.ss]*self.nn)
            array_y_part = array_y[:self.nn]
            correct = array_y_part[array_y_part == array_yhat_part].size
            gi += self.nn*(1-(correct/self.nn)**2)
        if self.nn < array_y.size: # right
            array_yhat_part = np.array([self.ss]*(array_y.size-self.nn))
            array_y_part = array_y[self.nn:]
            correct = array_y_part[array_y_part == array_yhat_part].size
            gi += (array_y.size-self.nn)*(1-(correct/(array_y.size-self.nn))**2)
        return gi

class DECISION_TREE:
    def __init__(self, max_h, hh, ll, rr, xy):
        self.left = None # left node
        self.right = None # right node
        self.max_height = max_h
        self.height = hh
        self.lb = ll # left bound
        self.rb = rr # right bound
        self.ds_ff = 0 # the feature for the decision stump
        self.ds_theta = 0 # theta
        self.ds_ss = 0 # s
        if xy.size > 0:
            self.train(xy)

    def train(self, xy):
        # sort the data by features
        num_features = xy.shape[0]-1
        sorted_xy = np.zeros((num_features, xy.shape[0], xy.shape[1]))
        list_ds = []
        data_size = xy.shape[1]
        for ff in range(num_features):
            sorted_xy[ff] = xy[:, xy[ff].argsort()]
        # initialize decision stumps
        for ss in range(-1, 2, 2): # the sign of a decision stump: -1 or 1
            list_ds.append(DECISION_STUMP(ss, 0))
            for nn in range(1, data_size):
                list_ds.append(DECISION_STUMP(ss, nn))
        # train
        pool = mp.Pool(processes=mp.cpu_count())
        min_ein = data_size*10
        min_eid = 0
        target_nn = 0
        for ff in range(num_features):
            sorted_xy[ff] = xy[:, xy[ff].argsort()]
            result = [pool.apply_async(list_ds[dd].get_gi, args=(sorted_xy[ff][-1], )) for dd in range(len(list_ds))]
            list_gid = [rr.get() for rr in result]
            ein = min(list_gid)
            if ein < min_ein:
                min_ein = ein
                min_eid = list_gid.index(min_ein)
                self.ds_ss = list_ds[min_eid].ss
                self.ds_ff = ff
                target_nn = list_ds[min_eid].nn
        pool.close()
        # update
        if target_nn == 0:
            self.ds_theta = (self.lb+sorted_xy[self.ds_ff][self.ds_ff][0])/2
        else:
            self.ds_theta = (sorted_xy[self.ds_ff][self.ds_ff][target_nn-1]+sorted_xy[self.ds_ff][self.ds_ff][target_nn])/2
            if self.max_height > self.height:
                # left
                xy_leaf = sorted_xy[self.ds_ff][:, :target_nn]
                self.add(DECISION_TREE(self.max_height, self.height+1, self.lb, sorted_xy[self.ds_ff][self.ds_ff][target_nn], xy_leaf))
                # right
                xy_leaf = sorted_xy[self.ds_ff][:, target_nn:]
                self.add(DECISION_TREE(self.max_height, self.height+1, sorted_xy[self.ds_ff][self.ds_ff][target_nn-1], self.rb, xy_leaf))
        
    def add(self, dtree):
        if not bool(self.left):
#            print("add left, height = {}".format(dtree.height))
            self.left = dtree
        elif not bool(self.right):
#            print("add right, height = {}".format(dtree.height))
            self.right = dtree

def traverse(dtree):
    queue = [dtree]
    while bool(queue):
        ptr = queue.pop(0)
        print("[h{}] x{}, theta = {}, s = {}".format(ptr.height, ptr.ds_ff, ptr.ds_theta, ptr.ds_ss))
        if bool(ptr.left):
            queue.append(ptr.left)
        else:
            print("no left child")
        if bool(ptr.right):
            queue.append(ptr.right)
        else:
            print("no right child")

def load_xy(list_line):
    matrix_xy = []
    for ii in range(len(list_line)):
        data = list_line[ii].strip().split()
        list_xy = list(map(float, data[:-1])) # x
        list_xy.append(int(data[-1]))
        matrix_xy.append(list_xy)
    return np.transpose(np.array(matrix_xy))

def main():
    # load data
    file_data = "hw3_train.dat"
    xy_train = None
    with open(file_data, 'r') as fr:
        list_line = fr.readlines()
        xy_train = load_xy(list_line)

    dtree = DECISION_TREE(100, 1, 0, 1, xy_train)
    traverse(dtree)
            


    file_data = "hw3_test.dat"
    
    return

if __name__ == "__main__":
    main()

