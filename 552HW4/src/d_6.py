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

test_data = feature.loc[feature['train_test'] == 0]
test_data.loc[(test_data['activities']
               == 0) | (test_data['activities'] == 1), 'activities'] = 0
test_data.loc[(test_data['activities']
               != 0) & (test_data['activities'] != 1), 'activities'] = 1
