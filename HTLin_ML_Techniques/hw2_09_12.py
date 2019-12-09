import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import BaggingClassifier

def load_xy(list_line):
    x = []
    y = []
    for line in list_line:
        data = line.strip().split()
        x.append(list(map(float, data[:-1])))
        y.append(float(data[-1]))
    return np.array(x), np.array(y)

def main():
    # load data
    file_lssvm = "hw2_lssvm_all.dat"
    size_train = 400
    with open(file_lssvm, 'r') as fr:
        list_line = fr.readlines()
        x_train, y_train = load_xy(list_line[:size_train])
        x_test, y_test = load_xy(list_line[size_train:])
    list_lambda = [0.05, 0.5, 5, 50, 500]
    ein = []
    eout = []

    # Q9, Q10: run regression
    # for lam in list_lambda:
    #     rcf = RidgeClassifier(alpha=lam)
    #     rcf.fit(x_train, y_train)
    #     Ein = 1-rcf.score(x_train, y_train)
    #     Eout = 1-rcf.score(x_test, y_test)
    #     ein.append(Ein)
    #     eout.append(Eout)
    # print("argminEin = {}, minEin_lambda = {}".format(min(ein), list_lambda[ein.index(min(ein))]))
    # print("argminEout = {}, minEout_lambda = {}".format(min(eout), list_lambda[eout.index(min(eout))]))
    
    # Q11, Q12: Bagging
    num_iter = 250
    for lam in list_lambda:
        rcf = RidgeClassifier(alpha=lam)
        bcf = BaggingClassifier(base_estimator=rcf, n_estimators=num_iter, n_jobs=-1, random_state=0)
        bcf.fit(x_train, y_train)
        Ein = 1-bcf.score(x_train, y_train)
        Eout = 1-bcf.score(x_test, y_test)
        ein.append(Ein)
        eout.append(Eout)
    print("argminEin = {}, minEin_lambda = {}".format(min(ein), list_lambda[ein.index(min(ein))]))
    print("argminEout = {}, minEout_lambda = {}".format(min(eout), list_lambda[eout.index(min(eout))]))

    return

if __name__ == "__main__":
    main()

