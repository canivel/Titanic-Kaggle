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
from sklearn import grid_search
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

# Create the random forest object which will include all the parameters
# for the fit

df_train = pd.DataFrame.from_csv("data/train.csv")
df_test = pd.DataFrame.from_csv("data/test.csv")
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

df_train = df_train.drop(['Sex', 'Embarked', 'Age'], axis=1)
df_train = df_train.dropna()


df_test = df_test.drop(['Sex', 'Embarked', 'Age'], axis=1)
df_test = df_test.dropna()

FEATURES = ['Pclass', 'SibSp', 'Parch', 'Fare', 'Gender', 'Port_of_Embarkation', 'AgeFill', 'AgeIsNull',
            'FamilySize', 'Age*Class']

features_train = np.array(df_train[FEATURES].values)
labels_train = df_train["Survived"]

features_test = np.array(df_test[FEATURES].values)


data_train = df_train
data_test = df_test

# features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3,
#                                                                           random_state=42)

#scale data
min_max_scaler = preprocessing.MinMaxScaler()
features_train_scaled = min_max_scaler.fit_transform(features_train)
features_test_scaled = min_max_scaler.fit_transform(features_test)

# forest = RandomForestClassifier(n_estimators=100)
# forest = forest.fit(features_train, labels_train)
# output = forest.predict(features_test)
#
# print accuracy_score(labels_test, output)


print "Decision Tree"
param_grid = {'criterion': ('gini', 'entropy'),
              'splitter': ('best', 'random'),
              'min_samples_split': [4, 5, 10, 20],
              'max_features': ('auto', 'sqrt', 'log2', None),
              'max_depth': [None, 1, 2, 10, 50],
              'max_leaf_nodes': [None, 8]}
clf = grid_search.GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid)

clf = clf.fit(features_train, labels_train)
print(clf.best_estimator_)
pred = clf.predict(features_test)
print len(pred)

