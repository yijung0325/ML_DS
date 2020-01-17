import numpy as np
import multiprocessing as mp
#import matplotlib.pyplot as plt

class DECISION_TREE:
    def __init__(self):
        self.left = None
        self.right = None

    def determine(self, ff, theta, ss):
        self.ds_ff = ff
        self.ds_theta = theta
        self.ds_ss = ss

    def add(self, tree):
        if not bool(self.left):
            self.left = tree
        elif not bool(self.right):
            self.right = tree

class DECISION_STUMP:
    def __init__(self, ss, nn, ff, array_x):
        self.ss = ss
        self.ff = ff # feature
        self.nn = nn
        self.array_yhat = np.append([-ss]*nn, [ss]*(array_x.size-nn))
        self.array_yhat = self.array_yhat.astype(int)

    def get_ein(self, array_y):
        Ein = array_y[array_y != self.array_yhat].size
        return Ein

def branching(dtree, xy):
    # sort the data by features, and initialize decision stumps
    num_features = xy.shape[0]-1
    sorted_xy = np.zeros((num_features, xy.shape[0], xy.shape[1]))
    list_ds = []
    data_size = xy.shape[1] 
    for ff in range(num_features):
        sorted_xy[ff] = xy[:, xy[ff].argsort()]
        for ss in range(-1, 2, 2): # the sign of a decision stump: -1 or 1
            list_ds.append(DECISION_STUMP(ss, 0, ff, sorted_xy[ff][ff]))
            for nn in range(1, data_size):
                list_ds.append(DECISION_STUMP(ss, nn, ff, sorted_xy[ff][ff]))
    pool = mp.Pool(processes=mp.cpu_count())
    min_ein = data_size*10
    min_eid = 0
    target_ss = 0
    target_ff = 0
    target_nn = 0
    for nn in range(num_features):
        sorted_xy[nn] = xy[:, xy[nn].argsort()]
        result = [pool.apply_async(list_ds[dd].get_ein, args=(sorted_xy[nn][-1], )) for dd in range(len(list_ds))]
        list_ein = [rr.get() for rr in result]
        ein = min(list_ein)
        if ein < min_ein:
            min_ein = ein
            min_eid = list_ein.index(min_ein)
            target_ss = list_ds[min_eid].ss
            target_ff = list_ds[min_eid].ff
            target_nn = list_ds[min_eid].nn
    pool.close()
    # update
    target_theta = None
    xy_left = None
    xy_right = None
    if target_nn == 0:
        target_theta = sorted_xy[target_ff][0]/2
    else:
        target_theta = (sorted_xy[target_ff][target_nn-1]+sorted_xy[target_ff][target_nn])/2
        xy_left = sorted_xy[target_ff][:, :target_nn]
        xy_right = sorted_xy[target_ff][:, target_nn:]
    dtree.determin(target_ff, target_theta, target_ss)
    return [xy_left, xy_right]

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

    dtree = DECISION_TREE()
    xy_leaf = branching(dtree, xy_train)
    for xy in xy_leaf:
        dtree_new = DECISION_TREE()
        if bool(xy):
            xy_leaf_new = branching(dtree_new, xy)
            
        


    list_xy = [xy_train]
    while bool(list_xy):
        for xy in list_xy:
            target_theta, target_ss, target_ff, target_nn = branching(xy_train)
        list_xy.clear()


    file_data = "hw3_test.dat"
    
    return

if __name__ == "__main__":
    main()

