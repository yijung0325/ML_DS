# https://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_twoclass.html

import matplotlib.pyplot as plt
import multiprocessing as mp

class DECISION_STUMP:
    def __init__(self, ss, ii, list_x, nn, size):
        self.ss = ss
        self.ii = ii
        if nn == 0:
            self.theta = 0
        elif nn == size:
            self.theta = 1
        else:
            self.theta = (list_x[nn-1]+list_x[nn])/2
        self.list_yhat = [-ss]*nn
        self.list_yhat.extend([ss]*(size-nn))

    def get_err(self, list_y, list_u):
        error = 0
        for pp in range(len(list_y)):
            if list_y[pp] != self.list_yhat[pp]:
                error += list_u[pp]
        error /= sum(list_u)
        return error

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
            list_xy.append((x[jj], y[jj]))
        list_xy = sorted(list_xy, key=lambda xy: xy[0][ii])
        matrix_xy.append(list_xy)
    return matrix_xy

def decision_stump(list_y, list_yhat, list_u):
    Ein = 0
    for pp in range(len(list_y)):
        if list_yhat[pp] != list_y[pp]:
            Ein += list_u[pp]
    Ein /= sum(list_u)
    return Ein

def main():
    # load data
    file_data = "hw2_adaboost_train.dat"
    xy_train = None
    with open(file_data, 'r') as fr:
        list_line = fr.readlines()
        xy_train = load_xy(list_line)
    num_iter = 300

    # Q13
    list_u = [1/len(xy_train[0])]*len(xy_train[0])
    list_s = [-1, 1]
    list_gt = []
    pool = mp.Pool(processes=mp.cpu_count())
    for tt in range(num_iter):
        list_ein = []
        
        
        
        decision_stump(xy_train, list_u)
    pool.close()
#    for tt in range(1, num_iter+1):
#        pool.apply_async(AdaBoostClassifier(base_estimator=dcf, n_estimators=tt))

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
        
    file_data = "hw2_adaboost_test.dat"
    xy_test = None
    with open(file_data, 'r') as fr:
        list_line = fr.readlines()
        xy_test = load_xy(list_line)
    
    
    return

if __name__ == "__main__":
    main()

