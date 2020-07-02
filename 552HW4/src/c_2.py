import csv
import pandas as pd
import numpy as np
import os
from extract_feature import ext_feature


path = "./raw_data"
feature = ext_feature(path, 1, 5)
feature.to_csv("./src/feature.csv", index=False)
