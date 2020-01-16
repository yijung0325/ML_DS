import numpy as np
#import multiprocessing as mp
#import matplotlib.pyplot as plt


def classify(sorted_xy, list_ds):
    return

def branching(xy):
    data_size = xy.shape[1]
    list_ds = []
    for ss in range(-1, 2, 2): # the sign of a decision stump: -1 or 1
        list_ds.append(np.full(data_size, ss))
        for nn in range(1, data_size):
            list_ds.append(np.append([-ss]*nn, [ss]*(data_size-nn)))
    # sort the data by features
    num_features = xy.shape[0]-1
    min_ein = 100
    target_theta = 0
    target_s = 0
    target_idf = 0
    for nn in range(num_features):
        ein, theta, ss = classify(xy[:, xy[nn].argsort()], list_ds)
        if ein < min_ein:
            target_theta = theta
            target_s = ss
            target_idf = nn
            ein = min_ein
    return

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

    branching(xy_train)    
    file_data = "hw3_test.dat"
    
    return

if __name__ == "__main__":
    main()

