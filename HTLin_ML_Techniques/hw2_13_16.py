# https://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_twoclass.html

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
import multiprocessing as mp

def load_xy(list_line):
    x = []
    y = []
    for line in list_line:
        data = line.strip().split()
        x.append(list(map(float, data[:-1])))
        y.append(float(data[-1]))
    # sort
    matrix_xy = []
    for ii in range(len(x[0])):
        list_xy = []
        for jj in range(len(y)):
            list_xy.append((x[jj][ii], y[jj]))
        list_xy = sorted(list_xy, key=lambda xy: xy[0])
        matrix_xy.append(list_xy)
    return np.array(matrix_xy)

def decision_stump(x, y, u):
    dcf = DecisionTreeClassifier(max_depth=1)
    dcf.fit(x, y)
    Ein = 1-dcf.score(x, y)
    return Ein

def main():
    # load data
    file_data = "hw2_adaboost_train.dat"
    x_train = None
    y_train = None
    with open(file_data, 'r') as fr:
        list_line = fr.readlines()
        x_train, y_train = load_xy(list_line)
    file_data = "hw2_adaboost_test.dat"
    x_test = None
    y_test = None
    with open(file_data, 'r') as fr:
        list_line = fr.readlines()
        x_test, y_test = load_xy(list_line)
    num_iter = 300

    # Q13
    pool = mp.Pool(processes=mp.cpu_count())
    for tt in range(1, num_iter+1):
        pool.apply_async(AdaBoostClassifier(base_estimator=dcf, n_estimators=tt))

    # list_ein = []
    # for tt in range(1, num_iter+1):
    #     dcf = DecisionTreeClassifier(max_depth=1)
    #     abcf = AdaBoostClassifier(base_estimator=dcf, n_estimators=tt)
    #     abcf.fit(x_train, y_train)
    #     Ein = 1-abcf.score(x_train, y_train)
    #     list_ein.append(Ein)
    plt.xlabel("t")
    plt.ylabel("Ein")
    plt.plot(list(range(1, num_iter+1)), list_ein)
    plt.savefig("t_ein.png")
        
    
    
    return

if __name__ == "__main__":
    main()

