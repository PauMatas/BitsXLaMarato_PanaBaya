import pandas as pd
import numpy as np
from scipy.stats import pearsonr, stats
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv("clean_data.csv")
diagnosis = pd.read_csv("diagnosis_data.csv")

def test_hipotesis(column, data):
    clean_list1, clean_list2 = [], []
    a, b = np.array(data[column]), np.array(diagnosis['final_diagnosis_code'])
    for i in range(len(a)):
        if not np.isnan(a[i]) and not np.isnan(b[i]):
            clean_list1.append(a[i])
            clean_list2.append(b[i])
    if not clean_list1 or np.all(clean_list1 == clean_list1[0]) or np.all(clean_list2 == clean_list2[0]): # casos en el cas no hi ha cap valor numeric o es te una array uniforme
        return False
    stat, pval = stats.pearsonr(clean_list1, clean_list2)
    print(column, pval)
    return pval < 0.05

impactful_variables = []
for column in data.columns:
    if (data.dtypes[column] in ['float64', 'int64']):
        if (test_hipotesis(column, data)):
            impactful_variables.append(column)

print(impactful_variables)