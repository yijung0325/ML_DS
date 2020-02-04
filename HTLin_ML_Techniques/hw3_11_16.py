# https://bit.ly/379wq2t

import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

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
            self.fit(xy)

    def fit(self, xy):
        # initialize decision stumps
        data_size = xy.shape[0]
        list_ds = []
        for ss in range(-1, 2, 2): # the sign of a decision stump: -1 or 1
            list_ds.append(DECISION_STUMP(ss, 0))
            for nn in range(1, data_size):
                list_ds.append(DECISION_STUMP(ss, nn))
        # train
        num_features = xy.shape[1]-1
        sorted_xy = np.zeros((num_features, xy.shape[0], xy.shape[1]))
        min_ein = data_size*10
        min_eid = 0
        target_nn = 0
        for ff in range(num_features):
            sorted_xy[ff] = xy[xy[:, ff].argsort()]
            list_gid = []
            for ds in list_ds:
                list_gid.append(ds.get_gi(sorted_xy[ff][:, -1]))
            ein = min(list_gid)
            if ein < min_ein:
                min_ein = ein
                min_eid = list_gid.index(min_ein)
                self.ds_ss = list_ds[min_eid].ss
                self.ds_ff = ff
                target_nn = list_ds[min_eid].nn
        # update
        if target_nn == 0:
            self.ds_theta = (self.lb+sorted_xy[self.ds_ff][0][self.ds_ff])/2
        else:
            self.ds_theta = (sorted_xy[self.ds_ff][target_nn-1][self.ds_ff]+sorted_xy[self.ds_ff][target_nn][self.ds_ff])/2
            if self.max_height > self.height:
                # left
                xy_leaf = sorted_xy[self.ds_ff][:target_nn]
                self.add(DECISION_TREE(self.max_height, self.height+1, self.lb, sorted_xy[self.ds_ff][target_nn][self.ds_ff], xy_leaf))
                # right
                xy_leaf = sorted_xy[self.ds_ff][target_nn:]
                self.add(DECISION_TREE(self.max_height, self.height+1, sorted_xy[self.ds_ff][target_nn-1][self.ds_ff], self.rb, xy_leaf))
        
    def add(self, dtree):
        if not bool(self.left):
            self.left = dtree
        elif not bool(self.right):
            self.right = dtree

    def predict(self, xx):
        if xx[self.ds_ff] <= self.ds_theta:
            if bool(self.left):
                return self.left.predict(xx)
            else:
                return -self.ds_ss
        else:
            if bool(self.right):
                return self.right.predict(xx)
            else:
                return self.ds_ss

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
    return np.array(matrix_xy)

def get_yhat(xy, dtree):
    array_yhat = np.zeros(xy.shape[0])
    for ii in range(xy.shape[0]):
        array_yhat[ii] = dtree.predict(xy[ii, :2])
    return array_yhat

def get_err(array_y, array_yhat):
    return array_y[array_y != array_yhat].size/array_y.size

def q1516_yhat(list_yhat):
    matrix_yhat = np.array(list_yhat)
    array_yhat = np.zeros(matrix_yhat.shape[1])
    for ii in range(matrix_yhat.shape[1]):
        unique, count = np.unique(matrix_yhat[:, ii], return_counts=True)
        dict_count = dict(zip(unique, count))
        if -1 in dict_count and 1 in dict_count:
            if dict_count[1] > dict_count[-1]:
                array_yhat[ii] = 1
            elif dict_count[1] < dict_count[-1]:
                array_yhat[ii] = -1
            else:
                array_yhat[ii] = 0
        elif -1 not in dict_count:
            array_yhat[ii] = 1
        else: # elif 1 not in dict_count:
            array_yhat[ii] = -1
    return array_yhat

def main():
    # load data
    file_data = "hw3_train.dat"
    xy_train = None
    with open(file_data, 'r') as fr:
        list_line = fr.readlines()
        xy_train = load_xy(list_line)
    file_data = "hw3_test.dat"
    xy_test = None
    with open(file_data, 'r') as fr:
        list_line = fr.readlines()
        xy_test = load_xy(list_line)
    
    # Q11
#    dtree = DECISION_TREE(xy_train.size, 1, 0, 1, xy_train)
#    traverse(dtree)

    # Q12
    # Ein = 0
    # for xy in xy_train:
    #     if xy[-1] != dtree.predict(xy[:2]):
    #         Ein += 1
    # Ein /= xy_train.shape[0]
    # Eout = 0
    # for xy in xy_test:
    #     if xy[-1] != dtree.predict(xy[:2]):
    #         Eout += 1
    # Eout /= xy_test.shape[0]
    # print("Ein = {}, Eout = {}".format(Ein, Eout))

    # Q13
    # list_height = [1, 2, 3, 4, 5]
    # list_ein = []
    # list_eout = []
    # for hh in list_height:
    #     dtree = DECISION_TREE(hh, 1, 0, 1, xy_train)
    #     Ein = 0
    #     for xy in xy_train:
    #         if xy[-1] != dtree.predict(xy[:2]):
    #             Ein += 1
    #     Ein /= xy_train.shape[0]
    #     list_ein.append(Ein)
    #     Eout = 0
    #     for xy in xy_test:
    #         if xy[-1] != dtree.predict(xy[:2]):
    #             Eout += 1
    #     Eout /= xy_test.shape[0]
    #     list_eout.append(Eout)
    # plt.plot(list_height, list_ein, label="Ein")
    # plt.plot(list_height, list_eout, label="Eout")
    # plt.legend()
    # plt.xlabel("height")
    # plt.savefig("height_err.png")

    # Q14 - Q16
    num_trees = 30000
    ratio = 0.8
    pool = mp.Pool(processes=mp.cpu_count())
    list_xy_sub = []
    for nn in range(num_trees):
        list_xy_sub.append(xy_train[np.random.choice(xy_train.shape[0], int(xy_train.shape[0]*ratio), replace=False)])
    list_result = [pool.apply_async(DECISION_TREE, args=(xy_sub.size, 1, 0, 1, xy_sub)) for xy_sub in list_xy_sub]
    list_gt = [rr.get() for rr in list_result]
    list_result = [pool.apply_async(get_yhat, args=(xy_train, list_gt[ii],)) for ii in range(len(list_gt))]
    list_yhat = [rr.get() for rr in list_result]

    # Q14
#    list_result = [pool.apply_async(get_err, args=(xy_train[:, -1], list_yhat[ii],)) for ii in range(len(list_yhat))]
#    list_ein = [rr.get() for rr in list_result]
#    plt.hist(list_ein)
#    plt.xlabel("Ein")
#    plt.ylabel("histogram")
#    plt.savefig("hist_ein_bag.png")

    # Q15 - Q16
    # Ein
    list_result = [pool.apply_async(q1516_yhat, args=(list_yhat[:tt+1], )) for tt in range(num_trees)]
    list_yhat_bag = [rr.get() for rr in list_result]
    list_result = [pool.apply_async(get_err, args=(xy_train[:, -1], list_yhat_bag[ii],)) for ii in range(len(list_yhat))]
    list_ein = [rr.get() for rr in list_result]
    # Eout
    list_result = [pool.apply_async(get_yhat, args=(xy_test, list_gt[ii],)) for ii in range(len(list_gt))]
    list_yhat = [rr.get() for rr in list_result]
    list_result = [pool.apply_async(q1516_yhat, args=(list_yhat[:tt+1], )) for tt in range(num_trees)]
    list_yhat_bag = [rr.get() for rr in list_result]
    list_result = [pool.apply_async(get_err, args=(xy_test[:, -1], list_yhat_bag[ii],)) for ii in range(len(list_yhat))]
    list_eout = [rr.get() for rr in list_result]
    # plot
    list_t = list(range(1, num_trees+1))
    plt.plot(list_t, list_ein, label="Ein")
    plt.plot(list_t, list_eout, label="Eout")
    plt.legend()
#    plt.xticks(list(range(1, num_trees+1, round(num_trees/20))))
    plt.xlabel("t")
    plt.ylabel("err rate")
    plt.savefig("q1516.png")

    pool.close()

    
    return

if __name__ == "__main__":
    main()

