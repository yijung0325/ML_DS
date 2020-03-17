import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def load_x(list_line):
    list_x = []
    for ii in range(len(list_line)):
        data = list_line[ii].strip().split()
        list_x.append(list(map(float, data[:-1]))) # x
    return np.array(list_x)

def main():
    # load data
    file_data = "hw4_nolabel_train.dat"
    x_train = None
    with open(file_data, 'r') as fr:
        list_line = fr.readlines()
        x_train = load_x(list_line)

    num_exp = 500
    list_k = [2, 4, 6, 8, 10]
    list_ein_avg = []
    list_ein_var = []
    for kk in list_k:
        Ein_avg = 0
        Ein_var = 0
        for nn in range(num_exp):
            kmeans = KMeans(n_clusters=kk, init="random")
            kmeans.fit(x_train)
            distance = kmeans.transform(x_train)
            labels = kmeans.labels_
            ein = 0
            for nn in range(x_train.shape[0]):
                ein += (distance[nn][labels[nn]])**2
            ein /= x_train.shape[0]
            Ein_avg += ein
            Ein_var += ein**2
        Ein_avg /= num_exp
        Ein_var = Ein_var/num_exp-Ein_avg**2
        list_ein_avg.append(Ein_avg)
        list_ein_var.append(Ein_var)
    plt.subplot(211)
    plt.plot(list_k, list_ein_avg, label="Ein_avg")
    plt.legend()
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.subplot(212)
    plt.plot(list_k, list_ein_var, label="Ein_var", color='r')
    plt.xlabel("k")
    plt.legend()
    plt.savefig("Ein_Kmeans.png")
    return

if __name__ == "__main__":
    main()

