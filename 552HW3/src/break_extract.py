import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import shutil


def brk_ext(piece):

    if os.path.exists("./breaking_data"):
        shutil.rmtree("./breaking_data")
    os.mkdir("./breaking_data")
    folders = os.listdir("./raw_data")
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
    q = 0
    for folder in folders:
        n = 0
        if folder.startswith("bending"):
            merge_index = 2
        else:
            merge_index = 3
        for f in os.listdir("./raw_data/"+folder):
            data = pd.read_csv("./raw_data/" + folder + "/" + f,
                               skiprows=5, header=None)

            new_data = pd.DataFrame()
            for j in range(0, int(480/piece)*piece, int(480/piece)):

                if not os.path.exists("./breaking_data/"+folder):
                    os.mkdir("./breaking_data/"+folder)
                p = pd.DataFrame(
                    data[j:j+int(480/piece)]).reset_index(drop=True)
                new_data = pd.concat(
                    [new_data, p], axis=1)
            new_data.columns = range(0, 7*piece)
            m = 0
            for k in range(0, 6*piece):
                if k % 6 == 0:
                    m += 1

                feature[i, k*7] = new_data.max()[k+m]
                feature[i, k*7+1] = new_data.min()[k+m]
                feature[i, k*7+2] = new_data.mean()[k+m]
                feature[i, k*7+3] = new_data.std()[k+m]
                feature[i, k*7+4] = new_data.quantile(.25)[k+m]
                feature[i, k*7+5] = new_data.median()[k+m]
                feature[i, k*7+6] = new_data.quantile(.75)[k+m]
            if n < merge_index:  # train data or test data
                feature[i, -2] = 0  # test
            else:
                feature[i, -2] = 1  # train
            feature[i, -1] = q  # activeties index
            n += 1
            i += 1
        q += 1
    feature = np.around(feature, 3)
    df = pd.DataFrame(feature, columns=col)
    return df
