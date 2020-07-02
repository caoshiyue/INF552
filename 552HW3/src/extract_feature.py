import csv
import pandas as pd
import numpy as np
import os


def ext_feature(path, piece, skiprows):

    folders = os.listdir(path)
    feature = np.zeros([88, (42*piece + 2)])
    col = []
    for i in range(0, 6*piece):
        col.append("max"+str(i+1))
        col.append("min"+str(i+1))
        col.append("mean"+str(i+1))
        col.append("std"+str(i+1))
        col.append("1st quart"+str(i+1))
        col.append("median"+str(i+1))
        col.append("3rd quart"+str(i+1))
    col.append("train_test")
    col.append("activities")

    i = 0
    p = 0
    for folder in folders:
        j = 0
        if folder.startswith("bending"):
            merge_index = 2
        else:
            merge_index = 3
        for f in os.listdir(path + "/"+folder):
            data = pd.read_csv(path + "/" + folder + "/" + f,
                               skiprows=skiprows, header=None)
            m = 0
            for k in range(0, 6*piece):
                if k % 6 == 0:
                    m += 1
                feature[i, k*7] = data.max()[k+m]
                feature[i, k*7+1] = data.min()[k+m]
                feature[i, k*7+2] = data.mean()[k+m]
                feature[i, k*7+3] = data.std()[k+m]
                feature[i, k*7+4] = data.quantile(.25)[k+m]
                feature[i, k*7+5] = data.median()[k+m]
                feature[i, k*7+6] = data.quantile(.75)[k+m]
            if j < merge_index:  # train data or test data
                feature[i, -2] = 0  # test
            else:
                feature[i, -2] = 1  # train
            feature[i, -1] = p  # activeties index
            i += 1
            j += 1
        p += 1
    feature = np.around(feature, 3)
    df = pd.DataFrame(feature, columns=col)
    return df
