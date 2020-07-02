import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from break_extract import brk_ext
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc

piece = 6
feature = brk_ext(piece)

train_data = feature.loc[feature['train_test'] == 1]
train_data.loc[(train_data['activities']
                == 0) | (train_data['activities'] == 1), 'activities'] = 0
train_data.loc[(train_data['activities']
                != 0) & (train_data['activities'] != 1), 'activities'] = 1

col = []
for i in range(0, 6*piece):
    col.append("max"+str(i+1))
    col.append("mean"+str(i+1))
    col.append("std"+str(i+1))
train_target = train_data.iloc[:, -1]
train_data = train_data[col]

model = LogisticRegression(max_iter=20)
clf = RFECV(model, step=1, cv=5, n_jobs=-1)
clf = clf.fit(train_data, train_target)
pured_data = train_data.iloc[:, clf.support_]
print(pured_data.columns)
