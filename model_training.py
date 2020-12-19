import pandas as pd
import numpy as np
from scipy.stats import pearsonr, stats
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold

data = pd.read_csv("clean_data.csv")
diagnosis = pd.read_csv("diagnosis_data.csv")
print(len(data))
print(len(diagnosis))
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
y = diagnosis
X = diagnosis.loc[:,impactful_variables]
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=1)


# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_val)
acc_log = round(logreg.score(X_train, y_train) * 100, 2)
print('logreg accuracy:', acc_log)