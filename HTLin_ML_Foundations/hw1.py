import numpy as np
import random

def XY(str_line):
    str_x, str_y = str_line.split('\t')
    list_x = str_x.split()
    array_data = [1]
    for x in list_x:
        array_data.append(float(x))
    array_data.append(float(str_y))
    return np.array(array_data)

def sign(value):
    if value > 0:
        return 1
    else:
        return -1

def main_pla(list_data, eta, MAX_TRIAL):
    array_w = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    next_ii = 0
    for ii in range(len(list_data)):
        data = list_data[ii]
        if sign(np.inner(array_w, data[:5])) != data[-1]:
            next_ii = ii
            break
        elif ii == len(list_data)-1:
            return array_w, 0
    num_update = 0
    no_error = False
    while num_update<MAX_TRIAL and no_error==False:
        data = list_data[next_ii]
        array_w += eta*data[-1]*data[:5]
        num_update += 1
        list_ii = list(range(next_ii+1, len(list_data)))
        if next_ii > 0:
            list_ii.extend(list(range(next_ii)))
        for jj in list_ii:
            test = list_data[jj]
            if sign(np.inner(array_w, test[:5])) != test[-1]:
                next_ii = jj
                break
            elif jj == list_ii[-1]:
                no_error = True
        if no_error == True:
            break
    return array_w, num_update

def main_pocket(list_data, MAX_TRIAL):
    array_w = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    array_wo = np.copy(array_w)
    next_ii = -1
    num_eo = 0
    for ii in range(len(list_data)):
        if sign(np.inner(array_w, list_data[ii][:5])) != list_data[ii][-1]:
            if num_eo == 0:
                next_ii = ii
            num_eo += 1
    if num_eo == 0:
        return array_wo, 0
    num_update = 0
    while num_update<MAX_TRIAL and num_eo>0:
        array_w += list_data[next_ii][-1]*list_data[next_ii][:5]
        num_update += 1
        # check mistakes
        list_ii = list(range(next_ii+1, len(list_data)))
        if next_ii > 0:
            list_ii.extend(list(range(next_ii)))
        num_e1 = 0
        for ii in list_ii:
            if sign(np.inner(array_w, list_data[ii][:5])) != list_data[ii][-1]:
                if num_e1 == 0:
                    next_ii = ii
                num_e1 += 1
        # update
        if num_e1 < num_eo:
            array_wo = np.copy(array_w)
            num_eo = num_e1
    return array_wo, num_update

if __name__ == "__main__":
    file_train = "hw1_18_train.dat"
    list_lines = []
    with open(file_train, 'r') as fr:
        list_lines = fr.readlines()
    list_train = []
    for str_line in list_lines:
        list_train.append(XY(str_line.strip()))

#    array_w, num_update = main_pla(list_train, 1, 5000)
#    array_w, num_update = main_pocket(list_train, 50)
#    print(num_update)

#    mean_update = 0
#    for ii in range(2000):
#        random.shuffle(list_train)
#        array_w, num_update = main_pla(list_train, 0.5, 5000)
#        mean_update += num_update
#        print("[{}] num_update = {}, sum_np = {}".format(ii, num_update, mean_update))
#    mean_update /= 2000
#    print(mean_update)

    file_test = "hw1_18_test.dat"
    list_lines = []
    with open(file_test, 'r') as fr:
        list_lines = fr.readlines()
    list_test = []
    for str_line in list_lines:
        list_test.append(XY(str_line.strip()))

    mean_error = 0
    for ii in range(2000):
        random.shuffle(list_train)
        array_w, num_update = main_pocket(list_train, 100)
#        array_w, num_update = main_pla(list_train, 1, 50)
        num_e = 0
        for data in list_test:
            if sign(np.inner(array_w, data[:5])) != data[-1]:
                num_e += 1
        mean_error += num_e
        print("[{}] num_update = {}, num_e = {}, sum_e = {}".format(ii, num_update, num_e, mean_error))
    mean_error /= 2000*len(list_test)
    print(mean_error)


    

