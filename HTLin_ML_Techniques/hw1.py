#!/usr/bin/env python3
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np

def plot_decision_region(X, y, classifier, resolution = 0.02):
    markers = ('x', 'o')
    colors = ("green", "red")
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha = 0.3)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, yv in enumerate(np.unique(y)):
        plt.scatter(x = X[y == yv, 0], y = X[y == yv, 1], c = colors[idx], marker = markers[idx], label = "y = {}".format(yv))
    plt.legend()
    return

def main():
    list_x = [[1, 0], [0, 1], [0, -1], [-1, 0], [0, 2], [0, -2], [-2, 0]]
    list_y = [-1, -1, -1, 1, 1, 1, 1]
    
    # Q1
#    list_z = []
#    for data in list_x:
#        list_z.append([2*(data[1]**2)-4*data[0]+2, data[0]**2-2*data[1]-3])
#    svm = SVC(kernel="linear")
#    svm.fit(np.array(list_z), np.array(list_y))
#    print("coef = {}\nintercept = {}".format(svm.coef_, svm.intercept_))
#    plot_decision_region(np.array(list_z), np.array(list_y), svm, resolution = 0.02)

    # Q2
    svm = SVC(kernel="poly", degree=2, gamma=1, coef0=1)
    svm.fit(np.array(list_x), np.array(list_y))
    aa = 0
    list_alpha = []
    for ii in range(len(list_x)):
        if aa==svm.support_.shape[0] or ii!=svm.support_[aa]:
            list_alpha.append(0)
        else:
            list_alpha.append(svm.dual_coef_[0][aa]*list_y[ii])
            aa += 1
    print("SV = {}, alpha = {}".format(svm.support_vectors_, list_alpha))
    plot_decision_region(np.array(list_x), np.array(list_y), svm, resolution = 0.02)

    # Q3
    coef_c = svm.intercept_[0]
    coef_x1 = 0
    coef_x2 = 0
    coef_x1x2 = 0
    coef_x1_2 = 0
    coef_x2_2 = 0
    for ii in range(len(list_x)):
        coef_c += list_y[ii]*list_alpha[ii]
        coef_x1 += list_y[ii]*list_alpha[ii]*list_x[ii][0]
        coef_x2 += list_y[ii]*list_alpha[ii]*list_x[ii][1]
        coef_x1x2 += list_y[ii]*list_alpha[ii]*list_x[ii][0]*list_x[ii][1]
        coef_x1_2 += list_y[ii]*list_alpha[ii]*(list_x[ii][0]**2)
        coef_x2_2 += list_y[ii]*list_alpha[ii]*(list_x[ii][1]**2)
    coef_x1 *= 2
    coef_x2 *= 2
    coef_x1x2 *= 2
    print("coef_c = {}".format(coef_c))
    print("coef_x1 = {}".format(coef_x1))
    print("coef_x2 = {}".format(coef_x2))
    print("coef_x1x2 = {}".format(coef_x1x2))
    print("coef_x1_2 = {}".format(coef_x1_2))
    print("coef_x2_2 = {}".format(coef_x2_2))
    
    
    return

if __name__ == "__main__":
    main()