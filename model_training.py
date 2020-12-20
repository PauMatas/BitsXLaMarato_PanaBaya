# Data frames
import pandas as pd
# Numeric
import numpy as np
# Machine Learning
from scipy.stats import pearsonr, stats
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier



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
    return pval < 0.05


def feature_selection(data):
    impactful_variables = []
    for column in data.columns:
        if (data.dtypes[column] in ['float64', 'int64']):
            if (test_hipotesis(column, data)):
                impactful_variables.append(column)
    return impactful_variables



y = diagnosis['final_diagnosis_code']
features = feature_selection(data)
X = data.loc[:,features]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=1)

print('features:')
print(features)
print()
print()
# N = len(X)
# X_train, X_val, y_train, y_val = X.iloc[:N//2, :], X.iloc[N//2:, :], y.iloc[:N//2], y.iloc[N//2:]

print(len(X_train))
print(len(X_val))




# Random Forest

random_forest = RandomForestClassifier(n_estimators=100, max_features='sqrt')
random_forest.fit(X_train, y_train)


from joblib import dump, load
dump(random_forest, 'model.joblib')



acc_random_forest = round(random_forest.score(X_val, y_val), 10)
print('Random forest accuracy:', acc_random_forest)

y_pred = random_forest.predict(X_val)
val_mae = mean_absolute_error(y_pred, y_val)
print('MAE:', val_mae)


scores = cross_val_score(random_forest, X_val, y_val, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



def get_model():
    return random_forest
