# -*- coding: utf-8 -*-

from sklearn import linear_model
from sklearn import model_selection
import numpy as np

def get_data(file_path):
    list_line = []
    with open(file_path, 'r') as fr:
        list_line = fr.readlines()
    list_x = []
    list_y = []
    for line in list_line:
        data = (line.strip()).split()
        list_x.append([])
        for x in data[:-1]:
            list_x[-1].append(float(x))
        list_y.append(float(data[-1]))
    return np.array(list_x), np.array(list_y)

def sign(x):
    if type(x) == np.ndarray:
        list_sign_x = []
        for xx in x:
            list_sign_x.append(sign(xx))
        return np.array(list_sign_x)
    else:
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0

def error(y_hat, y):
    error = 0
    for ii in range(y_hat.size):
        if y_hat[ii] != y[ii]:
            error += 1
    error /= y_hat.size
    return error

def ridge_reg(x_train, y_train, x_test, y_test, alphaa):
    reg = linear_model.Ridge(alpha=alphaa)
    reg.fit(x_train, y_train)
    Ein = error(sign(reg.predict(x_train)), y_train)
    Eout = error(sign(reg.predict(x_test)), y_test)
    return Ein, Eout

def main():
    x_train, y_train = get_data("hw4_train.dat")
    x_test, y_test = get_data("hw4_test.dat")

    # q13
    alpha = 10
    Ein, Eout = ridge_reg(x_train, y_train, x_test, y_test, alpha)
    print("Ein = {}, Eout = {}".format(Ein, Eout))

    # q14
#    list_alpha = list(range(-10, 3))
#    Ein = 10000
#    Eout = 10000
#    alpha = -100
#    for ii in range(len(list_alpha)):
#        Ein_a, Eout_a = ridge_reg(x_train, y_train, x_test, y_test, 10**list_alpha[ii])
#        if Ein_a <= Ein:
#            Ein = Ein_a
#            Eout = Eout_a
#            alpha = list_alpha[ii]
#    print("alpha = {}, Ein = {}, Eout = {}".format(alpha, Ein, Eout))

    # q15
#    Ein = 10000
#    Eout = 10000
#    alpha = -100
#    for ii in range(len(list_alpha)):
#        Ein_a, Eout_a = ridge_reg(x_train, y_train, x_test, y_test, 10**list_alpha[ii])
#        if Eout_a <= Eout:
#            Ein = Ein_a
#            Eout = Eout_a
#            alpha = list_alpha[ii]
#    print("alpha = {}, Ein = {}, Eout = {}".format(alpha, Ein, Eout))

    # q16
#    X_train, X_val, Y_train, Y_val = model_selection.train_test_split(x_train, y_train, test_size=0.4, shuffle=False)
#    Etrain = 10000
#    Eval = 10000
#    Eout = 10000
#    alpha = -100
#    for ii in range(len(list_alpha)):
#        reg = linear_model.Ridge(alpha=10**list_alpha[ii])
#        reg.fit(X_train, Y_train)
#        Etrain_a = error(sign(reg.predict(X_train)), Y_train)
#        Eval_a = error(sign(reg.predict(X_val)), Y_val)
#        Eout_a = error(sign(reg.predict(x_test)), y_test)
#        if Etrain_a <= Etrain:
#            Etrain = Etrain_a
#            Eval = Eval_a
#            Eout = Eout_a
#            alpha = list_alpha[ii]
#    print("alpha = {}, Etrain = {}, Eval = {}, Eout = {}".format(alpha, Etrain, Eval, Eout))

    # q17
#    for ii in range(len(list_alpha)):
#        reg = linear_model.Ridge(alpha=10**list_alpha[ii])
#        reg.fit(X_train, Y_train)
#        Etrain_a = error(sign(reg.predict(X_train)), Y_train)
#        Eval_a = error(sign(reg.predict(X_val)), Y_val)
#        Eout_a = error(sign(reg.predict(x_test)), y_test)
#        if Eval_a <= Eval:
#            Etrain = Etrain_a
#            Eval = Eval_a
#            Eout = Eout_a
#            alpha = list_alpha[ii]
#    print("alpha = {}, Etrain = {}, Eval = {}, Eout = {}".format(alpha, Etrain, Eval, Eout))

    # q18
#    alpha = 0
#    Ein, Eout = ridge_reg(x_train, y_train, x_test, y_test, 10**alpha)
#    print("Ein = {}, Eout = {}".format(Ein, Eout))

    # q19
#    num_split = 5
#    data_size = x_train.shape[0]
#    dict_ecv = {}
#    for alpha_a in list_alpha:
#        dict_ecv[alpha_a] = 0
#        for nn in range(num_split):
#            idx_s = nn*int(data_size/num_split)
#            idx_e = (nn+1)*int(data_size/num_split)
#            x_val = x_train[idx_s:idx_e][:]
#            y_val = y_train[idx_s:idx_e]
#            xt_train = np.concatenate((x_train[:idx_s][:], x_train[idx_e:][:]))
#            yt_train = np.append(y_train[:idx_s], y_train[idx_e:])
#            Ein, Ecv = ridge_reg(xt_train, yt_train, x_val, y_val, 10**alpha_a)
#            dict_ecv[alpha_a] += Ecv
#        dict_ecv[alpha_a] /= num_split
#    print(min(dict_ecv.items(), key=lambda x: x[1]))

    # q20
#    alpha = -8
#    Ein, Eout = ridge_reg(x_train, y_train, x_test, y_test, 10**alpha)
#    print("Ein = {}, Eout = {}".format(Ein, Eout))

    return

if __name__ == "__main__":
    main()
