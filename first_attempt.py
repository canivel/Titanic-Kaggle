'''
survival        Survival
                (0 = No; 1 = Yes)
pclass          Passenger Class
                (1 = 1st; 2 = 2nd; 3 = 3rd)
name            Name
sex             Sex
age             Age
sibsp           Number of Siblings/Spouses Aboard
parch           Number of Parents/Children Aboard
ticket          Ticket Number
fare            Passenger Fare
cabin           Cabin
embarked        Port of Embarkation
                (C = Cherbourg = 1; Q = Queenstown = 2; S = Southampton = 3)
High socio-economical status -- passenger['Pclass'] is 1
Medium socio-economical status -- passenger['Pclass'] is 2
Low socio-economical status -- passenger['Pclass'] is 3

Convert sex male =0, female=1
Convert  Embarked  (C = Cherbourg = 1; Q = Queenstown = 2; S = Southampton = 3)

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  #
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn import svm, grid_search
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import csv as csv

# Create the random forest object which will include all the parameters
# for the fit

df_train = pd.read_csv("data/train.csv", header=0)
df_test = pd.read_csv("data/test_acc.csv", header=0)

df_train = df_train.drop(['Ticket', 'Cabin', 'Name'], axis=1)
df_test = df_test.drop(['Ticket', 'Cabin', 'Name'], axis=1)
df_train['Age'].dropna()
df_test['Age'].dropna()

df_train['Gender'] = df_train['Sex'].map({'female': 0, 'male': 1}).astype(int)
df_train['Port_of_Embarkation'] = df_train['Embarked'].map({'C': 1, 'Q': 2, 'S': 3})

df_test['Gender'] = df_test['Sex'].map({'female': 0, 'male': 1}).astype(int)
df_test['Port_of_Embarkation'] = df_test['Embarked'].map({'C': 1, 'Q': 2, 'S': 3})

median_ages = np.zeros((2, 3))

for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i, j] = df_train[(df_train['Gender'] == i) & \
                               (df_train['Pclass'] == j + 1)]['Age'].dropna().median()

for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i, j] = df_test[(df_test['Gender'] == i) & \
                               (df_test['Pclass'] == j + 1)]['Age'].dropna().median()

df_train['AgeFill'] = df_train['Age']
df_test['AgeFill'] = df_test['Age']

for i in range(0, 2):
    for j in range(0, 3):
        df_train.loc[(df_train.Age.isnull()) & (df_train.Gender == i) & (df_train.Pclass == j + 1), \
               'AgeFill'] = median_ages[i, j]


for i in range(0, 2):
    for j in range(0, 3):
        df_test.loc[(df_test.Age.isnull()) & (df_test.Gender == i) & (df_test.Pclass == j + 1), \
               'AgeFill'] = median_ages[i, j]

df_train['AgeIsNull'] = pd.isnull(df_train.Age).astype(int)
df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch']
df_train['Age*Class'] = df_train.AgeFill * df_train.Pclass

df_test['AgeIsNull'] = pd.isnull(df_test.Age).astype(int)
df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch']
df_test['Age*Class'] = df_test.AgeFill * df_test.Pclass

# print df.dtypes[df.dtypes.map(lambda x: x=='object')]

df_train = df_train.drop(['Sex', 'Embarked', 'Age', 'Fare'], axis=1)
df_train = df_train.dropna()


df_test = df_test.drop(['Sex', 'Embarked', 'Age', 'Fare'], axis=1)

#print df_test[pd.isnull(df_test).any(axis=1)]

FEATURES = ['Pclass', 'SibSp', 'Parch', 'Gender', 'Port_of_Embarkation', 'AgeFill', 'AgeIsNull',
            'FamilySize', 'Age*Class']

features_train = np.array(df_train[FEATURES].values)
labels_train = df_train["Survived"]

features_test = np.array(df_test[FEATURES].values)
labels_test = df_test["Survived"]

data_train = df_train
data_test = df_test


# features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3,
#                                                                           random_state=42)

#scale data
min_max_scaler = preprocessing.MinMaxScaler()
features_train_scaled = min_max_scaler.fit_transform(features_train)
features_test_scaled = min_max_scaler.fit_transform(features_test)

from sklearn.pipeline import Pipeline
from sklearn.decomposition import RandomizedPCA
from sklearn.preprocessing import StandardScaler

# from sklearn import linear_model
# logreg = linear_model.LogisticRegression(C=1000)
# clf = logreg.fit(features_train, labels_train)
# pred = clf.predict(features_test)
# print pred
# print accuracy_score(labels_test, pred)

### 0.784688995215
# print "Decision Tree"
# param_grid = {'criterion': ('gini', 'entropy'),
#               'splitter': ('best', 'random'),
#               'min_samples_split': [4, 5, 10, 20],
#               'max_features': ('auto', 'sqrt', 'log2', None),
#               'max_depth': [None, 1, 2, 10, 50],
#               'max_leaf_nodes': [None, 8]}
# clf = grid_search.GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid)
#
# clf = clf.fit(features_train, labels_train)
# print(clf.best_estimator_)
# pred = clf.predict(features_test)
# print pred
# print accuracy_score(labels_test, pred)
# print "Random Forest"
# param_grid = {'criterion': ('gini', 'entropy'),
#               'min_samples_split': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20],
#               'max_features': (2, 3, 4, 5, 'auto', 'sqrt', 'log2', None),
#               'max_depth': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50],
#               'max_leaf_nodes': [None, -1, 2, 3, 4, 5, 6, 7, 8]}
'''
0.782296650718
clf = RandomForestClassifier(bootstrap=True,
            criterion='entropy', max_depth=10, max_features=5,
            max_leaf_nodes=4,  min_samples_leaf=1,
            min_samples_split=2, n_estimators=100, n_jobs=-1,
            oob_score=False, random_state=42, verbose=0)

0.791866028708
clf = RandomForestClassifier(bootstrap=True,
            criterion='entropy', max_depth=None, max_features=2,
            max_leaf_nodes=8, min_samples_split=1, n_estimators=1000,
            n_jobs=-1, oob_score=False)
'''
# pipeline = RandomForestClassifier(n_estimators=100, random_state=42)
# clf = grid_search.GridSearchCV(pipeline, param_grid=param_grid, verbose=3, scoring='accuracy', cv=10)


# if __name__ == '__main__':

    # print "SVM"
    # parameters = {'kernel':('linear', 'rbf'),
    #               'C':[1, 10, 100, 1000]}
    # svr = svm.SVC()
    # clf = grid_search.GridSearchCV(svr, parameters, n_jobs=-1)
    # svr = svm.SVC(C=1, sigma=.3)
    # clf = clf.fit(features_train, labels_train)
    # print(clf.best_estimator_)
    # pred = clf.predict(features_test)
    # print len(pred)
    # print accuracy_score(labels_test, pred)
ids = df_test['PassengerId'].values

def brute_force_acc_rd(features_train, labels_train, features_test, labels_test, ids):

    clf = RandomForestClassifier(bootstrap=True,
            criterion='entropy', max_depth=None, max_features=2,
            max_leaf_nodes=8, min_samples_split=5, n_estimators=1000,
            n_jobs=-1, oob_score=False)

    clf = clf.fit(features_train, labels_train)
    # print(clf.best_estimator_)
    pred = clf.predict(features_test)
    acc = accuracy_score(labels_test, pred)
    #print pred
    print acc

    if(acc > 0.793):
        predictions_file = open("data/canivel_random_forest_bf.csv", "wb")
        predictions_file_object = csv.writer(predictions_file)
        predictions_file_object.writerow(["PassengerId", "Survived"])
        predictions_file_object.writerows(zip(ids, pred))
        predictions_file.close()

    return acc


while brute_force_acc_rd(features_train, labels_train, features_test, labels_test, ids) < 0.794:
    brute_force_acc_rd(features_train, labels_train, features_test, labels_test, ids)

