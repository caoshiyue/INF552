import csv
import pandas as pd
import numpy as np
import heapq as hp

data = pd.read_csv("./src/feature.csv")

CI = []
for i in data.columns:
    samples = []
    for j in range(100):
        data_sample = data[i].sample(n=10, replace=True)
        sample_std = data_sample.std()
        hp.heappush(samples, sample_std)
    lower = np.around(hp.nsmallest(5, samples), 2)[4]
    higher = np.around(hp.nlargest(5, samples), 2)[4]
    CI.append([lower, higher])

table = pd.DataFrame({
    "STD": data.std()[:-2],
    "CI": CI[:-2]})
table
