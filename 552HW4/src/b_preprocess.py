import csv
import pandas as pd
import numpy as np
import os
import shutil

if os.path.exists("./merged_data"):
    shutil.rmtree("./merged_data")
os.mkdir("./merged_data")

folders = os.listdir("./raw_data")
for folder in folders:
    i = 0
    if folder.startswith("bending"):
        merge_index = 2
    else:
        merge_index = 3
    for f in os.listdir("./raw_data/"+folder):
        data = pd.read_csv("./raw_data/" + folder + "/" + f,
                           skiprows=5, header=None)

        if i < merge_index:
            outputfile = "./merged_data/" + folder+"_test.csv"
        else:
            outputfile = "./merged_data/" + folder+"_train.csv"
        data.to_csv(outputfile, mode='a', index=False, header=False)
        i += 1
