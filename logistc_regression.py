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
import re
import matplotlib.pyplot as plt  #
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import svm, grid_search
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import csv as csv
import string
# Create the random forest object which will include all the parameters
# for the fit
from dataframe_builder import build_dataframes


df_train, df_test = build_dataframes()


# FEATURES = ['Pclass', 'SibSp', 'Parch', 'Gender', 'Port_of_Embarkation', 'AgeFullFill', 'TicketNumber', 'TicketNumberDigits', 'TicketNumberStart',
#             'FamilySize', 'Age*Class', 'Deck_N', 'Title_N', 'Fare_Per_Person', 'AgeName', 'HighLow']

FEATURES = ['Pclass', 'FamilySize*Gender', 'Gender', 'AgeFullFill',  'TicketNumber', 'Gender*TicketNumberStart',
            'Age*Class', 'Deck_N', 'Title_N', 'Fare_Per_Person', 'AgeFullFill*HighLow', 'FamilyType']

features_train = np.array(df_train[FEATURES].values)
labels_train = df_train["Survived"]

features_test = np.array(df_test[FEATURES].values)
labels_test = df_test["Survived"]

data_train = df_train
data_test = df_test

#scale data
# min_max_scaler = preprocessing.MinMaxScaler()
# features_train_scaled = min_max_scaler.fit_transform(features_train)
# features_test_scaled = min_max_scaler.fit_transform(features_test)

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# features_train_scaled = scaler.fit_transform(features_train)
# features_test_scaled = scaler.fit_transform(features_test)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import svm, grid_search

# param_grid = {'penalty': ('l1', 'l2'),
#               'C': [1, 100, 1000]}
#
# clf = grid_search.GridSearchCV(LogisticRegression(), param_grid)

clf = LogisticRegression(C=1000, class_weight=None, dual=False, fit_intercept=True,
                         intercept_scaling=1, penalty='l1', random_state=None, tol=0.0001)

clf = clf.fit(features_train, labels_train)
# print(clf.best_estimator_)
pred = clf.predict(features_test)
print pred
print accuracy_score(labels_test, pred)

## 0.775119617225

