import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from break_extract import brk_ext
import statsmodels.api as sm
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc, roc_auc_score

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

model = LogisticRegression(max_iter=20)
clf = RFECV(model, step=1, cv=5, n_jobs=-1)
clf = clf.fit(pured_data, train_target)
pred = clf.predict(pured_data)
falsePositiveRate, truePositiveRate, thresholds = roc_curve(
    train_target, pred)

confu_mat = pd.crosstab(train_target, pred,
                        rownames=['True'], colnames=['Predicted'], margins=True)
print("confusion matrix")
print(confu_mat)
print("parameters")
statLogitModel = sm.Logit(train_target, pured_data).fit_regularized()
print(statLogitModel.params)
print("P-values")
scores, pvalues = chi2(pured_data, train_target)
for i in range(len(pvalues)):
    print(pured_data.columns[i], pvalues[i])
plt.figure(figsize=(16, 9))
plt.plot(falsePositiveRate, truePositiveRate)
plt.plot([0, 1], [0, 1], linestyle='dotted')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC and AUC')
plt.show()

test_data = feature.loc[feature['train_test'] == 0]
test_data.loc[(test_data['activities']
               == 0) | (test_data['activities'] == 1), 'activities'] = 0
test_data.loc[(test_data['activities']
               != 0) & (test_data['activities'] != 1), 'activities'] = 1
test_target = test_data.iloc[:, -1]
test_data = test_data[pured_data.columns]
accuracy = clf.score(test_data, test_target)

print("accuracy: %.2f" % accuracy)
