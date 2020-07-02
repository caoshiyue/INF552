import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import shutil


def brk_data(piece):

    if os.path.exists("./breaking_data"):
        shutil.rmtree("./breaking_data")
    os.mkdir("./breaking_data")
    folders = os.listdir("./raw_data")

    i = 0
    for folder in folders:
        for f in os.listdir("./raw_data/"+folder):
            data = pd.read_csv("./raw_data/" + folder + "/" + f,
                               skiprows=5, header=None)
            i += 1
            new_data = pd.DataFrame()
            for j in range(0, 480, int(480/piece)):

                if not os.path.exists("./breaking_data/"+folder):
                    os.mkdir("./breaking_data/"+folder)
                outputfile = "./breaking_data/"+folder+"/"+str(i)+".csv"
                p = pd.DataFrame(
                    data[j:j+int(480/piece)]).reset_index(drop=True)
                new_data = pd.concat(
                    [new_data, p], axis=1)

            new_data.to_csv(outputfile, index=False,
                            header=False)
