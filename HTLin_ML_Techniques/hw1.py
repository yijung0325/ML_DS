#!/usr/bin/env python3
import matplotlib.pyplot as plt
from sklearn.svm import SVC
#from sklearn.svm import LinearSVC
#from sklearn.model_selection import KFold
#from sklearn.model_selection import train_test_split
import numpy as np

def plot_decision_region(X, y, classifier):
    markers = ('x', 'o')
    colors = ("green", "red")
    x1_min = X[:, 0].min()-1
    x1_max = X[:, 0].max()+1
    x2_min = X[:, 1].min()-1
    x2_max = X[:, 1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, (x1_max-x1_min)/1000), np.arange(x2_min, x2_max, (x2_max-x2_min)/1000))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha = 0.3)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, yv in enumerate(np.unique(y)):
        plt.scatter(x = X[y == yv, 0], y = X[y == yv, 1], c = colors[idx], marker = markers[idx], label = "y = {}".format(yv))
    plt.legend()
    return

def get_XY(file_data, target):
    list_x = []
    list_y = []
    with open(file_data, 'r') as fr:
        list_lines = fr.readlines()
        for line in list_lines:
            list_data = (line.strip()).split()
            list_x.append([float(list_data[1]), float(list_data[2])])
            if target == int(float(list_data[0])):
                list_y.append(1)
            else:
                list_y.append(0)
    return np.array(list_x), np.array(list_y)

def error(svm, x, y):
    error = 0
    y_hat = svm.predict(x)
    error = y_hat[y_hat != y].size/y.shape[0]
    return round(error, 5)

def main():
    # Q1 - Q3
#    list_x = [[1, 0], [0, 1], [0, -1], [-1, 0], [0, 2], [0, -2], [-2, 0]]
#    list_y = [-1, -1, -1, 1, 1, 1, 1]

    # Q1
#    list_z = []
#    for data in list_x:
#        list_z.append([2*(data[1]**2)-4*data[0]+2, data[0]**2-2*data[1]-3])
#    svm = SVC(kernel="linear")
#    svm.fit(np.array(list_z), np.array(list_y))
#    print("coef = {}\nintercept = {}".format(svm.coef_, svm.intercept_))
#    plot_decision_region(np.array(list_z), np.array(list_y), svm, resolution = 0.02)

    # Q2
#    svm = SVC(kernel="poly", degree=2, gamma=1, coef0=1)
#    svm.fit(np.array(list_x), np.array(list_y))
#    aa = 0
#    list_alpha = []
#    for ii in range(len(list_x)):
#        if aa==svm.support_.shape[0] or ii!=svm.support_[aa]:
#            list_alpha.append(0)
#        else:
#            list_alpha.append(svm.dual_coef_[0][aa]*list_y[ii])
#            aa += 1
#    print("SV = {}, alpha = {}".format(svm.support_vectors_, list_alpha))
#    plot_decision_region(np.array(list_x), np.array(list_y), svm, resolution = 0.02)

    # Q3
#    coef_c = svm.intercept_[0]
#    coef_x1 = 0
#    coef_x2 = 0
#    coef_x1x2 = 0
#    coef_x1_2 = 0
#    coef_x2_2 = 0
#    for ii in range(len(list_x)):
#        coef_c += list_y[ii]*list_alpha[ii]
#        coef_x1 += list_y[ii]*list_alpha[ii]*list_x[ii][0]
#        coef_x2 += list_y[ii]*list_alpha[ii]*list_x[ii][1]
#        coef_x1x2 += list_y[ii]*list_alpha[ii]*list_x[ii][0]*list_x[ii][1]
#        coef_x1_2 += list_y[ii]*list_alpha[ii]*(list_x[ii][0]**2)
#        coef_x2_2 += list_y[ii]*list_alpha[ii]*(list_x[ii][1]**2)
#    coef_x1 *= 2
#    coef_x2 *= 2
#    coef_x1x2 *= 2
#    print("coef_c = {}".format(coef_c))
#    print("coef_x1 = {}".format(coef_x1))
#    print("coef_x2 = {}".format(coef_x2))
#    print("coef_x1x2 = {}".format(coef_x1x2))
#    print("coef_x1_2 = {}".format(coef_x1_2))
#    print("coef_x2_2 = {}".format(coef_x2_2))

    # Q13 - Q16
    file_data = "hw1_features.train"
    file_data = "hw1_features.test"

    # Q13
#    target = 2
#    x_train, y_train = get_XY(file_data, target)
#    x_test, y_test = get_XY(file_data, target)
#    list_C = [10**-5, 10**-3, 10**-1, 10**1, 10**3]
#    list_norm_w = []
#    for cc in list_C:
#        linear_svm = SVC(kernel="linear", C=cc, max_iter=100000000)
#        linear_svm.fit(x_train, y_train)
#        Ein = error(linear_svm, x_train, y_train)
#        Eout = error(linear_svm, x_test, y_test)
#        array_w = np.round(linear_svm.coef_, 5)
#        list_norm_w.append(round(np.linalg.norm(array_w), 5))
#        plot_decision_region(np.array(x_train), np.array(y_train), linear_svm)
#        print("C = {}: ||w|| = {}, Ein = {}, Eout = {}".format(cc, list_norm_w[-1], Ein, Eout))
#        print("w = {}, b = {}".format(linear_svm.coef_, linear_svm.intercept_))
#    plt.xscale("log")
#    plt.xlim(left=list_C[0], right=list_C[-1])
#    plt.xlabel("logC")
#    plt.ylabel("||w||")
#    plt.plot(list_C, list_norm_w)

    # Q14
#    target = 4
#    x_train, y_train = get_XY(file_data, target)
#    x_test, y_test = get_XY(file_data, target)
#    list_C = [10**-5, 10**-3, 10**-1, 10**1, 10**3]
#    list_ein = []
#    for cc in list_C:
#        poly_svm = SVC(kernel="poly", C=cc, gamma="auto", max_iter=100000000)
#        poly_svm.fit(x_train, y_train)
#        Ein = error(poly_svm, x_train, y_train)
#        Eout = error(poly_svm, x_test, y_test)
#        list_ein.append(Ein)
#        print("C = {}: Ein = {}, Eout = {}".format(cc, Ein, Eout))
#        plot_decision_region(np.array(x_train), np.array(y_train), poly_svm)
#    plt.xscale("log")
#    plt.xlim(left=list_C[0], right=list_C[-1])
#    plt.xlabel("logC")
#    plt.ylabel("Ein")
#    plt.plot(list_C, list_ein)

    # Q15
    target = 0
    x_train, y_train = get_XY(file_data, target)
    x_test, y_test = get_XY(file_data, target)
    list_C = [10**-2, 10**-1, 10**0, 10**1, 10**2]
    list_d = []
    for cc in list_C:
        rbf_svm = SVC(kernel="rbf", C=cc, gamma=80, max_iter=100000000)
        rbf_svm.fit(x_train, y_train)
        Ein = error(rbf_svm, x_train, y_train)
        Eout = error(rbf_svm, x_test, y_test)
        index = 0
        for ii in range((rbf_svm.support_).size):
            if abs(rbf_svm.dual_coef_[0][ii]) < cc: # free SV
                index = rbf_svm.support_[ii]
                break
        distance = round(abs(rbf_svm.decision_function(x_train[index].reshape(-1, x_train[index].shape[0]))[0]), 5)
        list_d.append(distance)
        print("C = {}: Ein = {}, Eout = {}, distance = {}".format(cc, Ein, Eout, distance))
#        plot_decision_region(np.array(x_train), np.array(y_train), rbf_svm)
    plt.xscale("log")
    plt.xlim(left=list_C[0], right=list_C[-1])
    plt.ylim(top=max(list_d)+0.1, bottom=min(list_d)-0.1)
    plt.xlabel("logC")
    plt.ylabel("distance")
    plt.plot(list_C, list_d)

#    kf = KFold(n_splits=5, random_state=123)
#    for index_train, index_val in kf.split(list_t_train):
#        print("Train: {}, Validate: {}".format(index_train, index_val))


#    x_train, x_val, y_train, y_val =  train_test_split(x_train, y_train, test_size = 0.2)

    
    return

if __name__ == "__main__":
    main()