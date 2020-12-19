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
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import mean_absolute_error



path = './clean_dataset'
df = pd.read_csv(path)

y = df['final_diagnosis_code']
features = df.columns
X = home_data[features]


X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=1)


# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_val)
acc_log = round(logreg.score(X_train, y_train) * 100, 2)
print('logreg accuracy:', acc_log)


# Support Vector Machines
svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_val)
acc_svc = round(svc.score(X_train, y_train) * 100, 2)
print('svc accuracy:', acc_svc)



# K Neasrest Neighbors
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_val)
acc_knn = round(knn.score(X_train, y_train) * 100, 2)
print('knn accuracy:', acc_knn)




# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
y_pred = gaussian.predict(X_val)
acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)
print('GaussianNB accuracy:', acc_gaussian)




# Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, y_train)
y_pred = perceptron.predict(X_val)
acc_perceptron = round(perceptron.score(X_train, y_train) * 100, 2)
print('Perceptron accuracy:', acc_perceptron)



# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, y_train)
y_pred = linear_svc.predict(X_val)
acc_linear_svc = round(linear_svc.score(X_train, y_train) * 100, 2)
print('Linear SVC accuracy:', acc_linear_svc)




# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, y_train)
y_pred = sgd.predict(X_val)
acc_sgd = round(sgd.score(X_train, y_train) * 100, 2)
print('SGDC accuracy:', acc_sgd)


# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_val)
acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)
print('Decission Tree accuracy:', acc_decision_tree)



# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_val)
random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
print('Random forest accuracy:', acc_random_forest)


# Model evaluation:
models = {
    'Support Vector Machines' : acc_svc,
    'KNN' : acc_knn,
    'Logistic Regression' : acc_log,
    'Random Forest' : acc_random_forest,
    'Naive Bayes' : acc_gaussian,
    'Perceptron' : acc_perceptron,
    'Stochastic Gradient Decent' : acc_sgd,
    'Linear SVC' : acc_linear_svc,
    'Decision Tree' : acc_decision_tree
}

sort_models = sorted(models.items(), key=lambda x: x[1], reverse=True)

bestModel = sort_models[0][0]
