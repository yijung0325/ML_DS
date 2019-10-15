# -*- coding: utf-8 -*-
import random
import numpy as np
import multiprocessing as mp
from sklearn.linear_model import LinearRegression

def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

def fun(x):
    return sign(x[0]*x[0]+x[1]*x[1]-0.6)

def gen_xy(data_size, ratio_flip):
    x_train = []
    y_train = []
    for ii in range(data_size):
        x1 = random.uniform(-1, 1)
        while x1 == 0:
            x1 = random.uniform(-1, 1)
        x2 = random.uniform(-1, 1)
        while x2 == 0:
            x2 = random.uniform(-1, 1)
        x_train.append([x1, x2])
        y_train.append(fun([x1, x2]))
    id_flip = []
    for ii in range(int(data_size*ratio_flip)):
        idd = random.randrange(data_size)
        while idd in id_flip:
            idd = random.randrange(data_size)
        id_flip.append(idd)
        y_train[idd] *= -1
    return x_train, y_train

def fit_err(x, y, w, w0):
    y_hat = sign(np.dot(x, w)+w0)
    if y == y_hat:
        return 0
    else:
        return 1

def q13_exp(kk, data_size, ratio_flip):
    # generate x_train
    x_train, y_train = gen_xy(data_size, ratio_flip)
    # linear regression
    reg = LinearRegression().fit(x_train, y_train)
    w = reg.coef_
    w0 = reg.intercept_
    # calculate Ein
    output = []
    for ii in range(data_size):
        output.append(fit_err(x_train[ii], y_train[ii], w, w0))
    Ein = sum(output)/data_size
    return w, w0, Ein

def q14_gg(ii, x):
    if ii == 0:
        return sign(-1-1.5*x[0]+0.08*x[1]+0.13*x[0]*x[1]+0.05*x[0]*x[0]+1.5*x[1]*x[1])
    elif ii == 1:
        return sign(-1-0.05*x[0]+0.08*x[1]+0.13*x[0]*x[1]+1.5*x[0]*x[0]+1.5*x[1]*x[1])
    elif ii == 2:
        return sign(-1-1.5*x[0]+0.08*x[1]+0.13*x[0]*x[1]+0.05*x[0]*x[0]+0.05*x[1]*x[1])
    elif ii == 3:
        return sign(-1-0.05*x[0]+0.08*x[1]+0.13*x[0]*x[1]+15*x[0]*x[0]+1.5*x[1]*x[1])
    else:
        return sign(-1-0.05*x[0]+0.08*x[1]+0.13*x[0]*x[1]+1.5*x[0]*x[0]+15*x[1]*x[1])

def transform(x):
    list_xt = []
    for ii in range(len(x)):
        list_xt.append([1, x[ii][0], x[ii][1], x[ii][0]*x[ii][1], x[ii][0]*x[ii][0], x[ii][1]*x[ii][1]])
    return np.array(list_xt)

def q14_exp(kk, data_size, ratio_flip):
    # generate x_train, y_train, and transforms
    x_train, y_train = gen_xy(data_size, ratio_flip)
    xt_train = transform(x_train)
    # linear regression
    reg = LinearRegression().fit(xt_train, y_train)
    w = reg.coef_
    w0 = reg.intercept_
    num_gg = 5
    list_gg = []
    for ii in range(num_gg):
        list_gg.append([])
    for ii in range(data_size):
        output = sign(np.dot(xt_train[ii], w)+w0)
        for gg in range(num_gg):
            if output == q14_gg(gg, x_train[ii]):
                list_gg[gg].append(1)
            else:
                list_gg[gg].append(0)
    Ein_gg = []
    for ii in range(num_gg):
        Ein_gg.append(sum(list_gg[ii])/data_size)
    return Ein_gg.index(max(Ein_gg))

def q15_exp(kk, data_size, ratio_flip):
    # generate x_train, y_train and transforms
    x_train, y_train = gen_xy(data_size, ratio_flip)
    xt_train = transform(x_train)
    # linear regression
    reg = LinearRegression().fit(xt_train, y_train)
    w = reg.coef_
    w0 = reg.intercept_
    # generate x_test, y_test
    x_test, y_test = gen_xy(data_size, ratio_flip)
    xt_test = transform(x_test)
    list_eout = []
    for ii in range(data_size):
        list_eout.append(fit_err(xt_test[ii], y_test[ii], w, w0))
    Eout = sum(list_eout)/data_size
    return Eout

def main():
    data_size = 1000
    ratio_flip = 0.1
    num_exp = 1000

    # Q13
#    pool = mp.Pool(processes=mp.cpu_count())
#    results = [pool.apply_async(q13_exp, args=(kk, data_size, ratio_flip,)) for kk in range(num_exp)]
#    output = [p.get() for p in results]
#    Ein = sum(output)/num_exp
#    print(Ein)

    # Q14
    pool = mp.Pool(processes=mp.cpu_count())
    results = [pool.apply_async(q14_exp, args=(kk, data_size, ratio_flip,)) for kk in range(num_exp)]
    output = [p.get() for p in results]
    output = np.array(output)
    counts = np.bincount(output)
    print(counts)
    print(np.argmax(counts)+1)

    # Q15
#    pool = mp.Pool(processes=mp.cpu_count())
#    results = [pool.apply_async(q15_exp, args=(kk, data_size, ratio_flip,)) for kk in range(num_exp)]
#    output = [p.get() for p in results]
#    Eout = sum(output)/num_exp
#    print(Eout)
    return

if __name__ == "__main__":
    main()