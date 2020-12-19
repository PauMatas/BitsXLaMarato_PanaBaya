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

################################################################################
import numpy as np
import pandas as pd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error


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
    #print(column, pval)
    return pval < 0.05

impactful_variables = []
for column in data.columns:
    if (data.dtypes[column] in ['float64', 'int64']):
        if (test_hipotesis(column, data)):
            impactful_variables.append(column)

# print(impactful_variables)



y = diagnosis['final_diagnosis_code']
X = data.loc[:,impactful_variables]
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=1)
# N = len(X)
# X_train, X_val, y_train, y_val = X.iloc[:N//2, :], X.iloc[N//2:, :], y.iloc[:N//2], y.iloc[N//2:]

print(len(X_train))
print(len(X_val))
print(len(y_train))
print(len(y_val))




# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)

# random_forest = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)
# scores = cross_val_score(random_forest, X_val, y_val, cv=5)
# print(scores.mean()*100)
#
# print('-----------------------------------------------------------------------------------------------')

random_forest.fit(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 10)
print('Random forest accuracy:', acc_random_forest)

y_pred = random_forest.predict(X_val)
val_mae = mean_absolute_error(y_pred, y_val)
print('MAE:', val_mae)



# # Logistic Regression
# logreg = LogisticRegression()
# logreg.fit(X_train, y_train)
# y_pred = logreg.predict(X_val)
# acc_log = round(logreg.score(X_train, y_train) * 100, 2)
# print('logreg accuracy:', acc_log)
#
#
# # Support Vector Machines
# svc = SVC()
# svc.fit(X_train, y_train)
# y_pred = svc.predict(X_val)
# acc_svc = round(svc.score(X_train, y_train) * 100, 2)
# print('svc accuracy:', acc_svc)
#
#
#
# # K Neasrest Neighbors
# knn = KNeighborsClassifier(n_neighbors = 3)
# knn.fit(X_train, y_train)
# y_pred = knn.predict(X_val)
# acc_knn = round(knn.score(X_train, y_train) * 100, 2)
# print('knn accuracy:', acc_knn)
#
#
#
#
# # Gaussian Naive Bayes
# gaussian = GaussianNB()
# gaussian.fit(X_train, y_train)
# y_pred = gaussian.predict(X_val)
# acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)
# print('GaussianNB accuracy:', acc_gaussian)
#
#
#
#
# # Perceptron
# perceptron = Perceptron()
# perceptron.fit(X_train, y_train)
# y_pred = perceptron.predict(X_val)
# acc_perceptron = round(perceptron.score(X_train, y_train) * 100, 2)
# print('Perceptron accuracy:', acc_perceptron)
#
#
#
# # Linear SVC
# linear_svc = LinearSVC()
# linear_svc.fit(X_train, y_train)
# y_pred = linear_svc.predict(X_val)
# acc_linear_svc = round(linear_svc.score(X_train, y_train) * 100, 2)
# print('Linear SVC accuracy:', acc_linear_svc)
#
#
#
#
# # Stochastic Gradient Descent
# sgd = SGDClassifier()
# sgd.fit(X_train, y_train)
# y_pred = sgd.predict(X_val)
# acc_sgd = round(sgd.score(X_train, y_train) * 100, 2)
# print('SGDC accuracy:', acc_sgd)
#
#
# # Decision Tree
# decision_tree = DecisionTreeClassifier()
# decision_tree.fit(X_train, y_train)
# y_pred = decision_tree.predict(X_val)
# acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)
# print('Decission Tree accuracy:', acc_decision_tree)
#
#
#


# Model evaluation:
# models = {
#     'Support Vector Machines' : acc_svc,
#     'KNN' : acc_knn,
#     'Logistic Regression' : acc_log,
#     'Random Forest' : acc_random_forest,
#     'Naive Bayes' : acc_gaussian,
#     'Perceptron' : acc_perceptron,
#     'Stochastic Gradient Decent' : acc_sgd,
#     'Linear SVC' : acc_linear_svc,
#     'Decision Tree' : acc_decision_tree
# }
#
# sort_models = sorted(models.items(), key=lambda x: x[1], reverse=True)
# print(sort_models)
# bestModel = sort_models[0][0]
