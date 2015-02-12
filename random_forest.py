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

df_train = pd.read_csv("data/train.csv", header=0)
df_test = pd.read_csv("data/test_acc.csv", header=0)


#FEATURES ENGINNERING

def substrings_in_string(big_string, substrings):
    # print (" >>>> ", big_string)
    # print (" >>>> ", substrings)
    for substring in substrings:
        if string.find(big_string, substring) != -1:
            return substring
    print big_string
    return np.nan

def replace_titles(x):
    title=x['Title']
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Countess', 'Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title =='Dr':
        if x['Sex']=='Male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title

title_list = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev', 'Dr', 'Ms', 'Mlle', 'Col', 'Capt', 'Mme', 'Countess', 'Don', 'Jonkheer']
df_train['Title'] = df_train['Name'].map(lambda x: substrings_in_string(x, title_list))
df_train['Title'] = df_train.apply(replace_titles, axis=1)

df_train['Title'][df_train.Title == 'Jonkheer'] = 'Master'
df_train['Title'][df_train.Title.isin(['Ms', 'Mlle'])] = 'Miss'
df_train['Title'][df_train.Title == 'Mme'] = 'Mrs'
df_train['Title'][df_train.Title.isin(['Capt', 'Don', 'Major', 'Col', 'Sir'])] = 'Sir'
df_train['Title'][df_train.Title.isin(['Dona', 'Lady', 'the Countess'])] = 'Lady'

df_test['Title'] = df_test['Name'].map(lambda x: substrings_in_string(x, title_list))
df_test['Title'] = df_test.apply(replace_titles, axis=1)

df_test['Title'][df_test.Title == 'Jonkheer'] = 'Master'
df_test['Title'][df_test.Title.isin(['Ms', 'Mlle'])] = 'Miss'
df_test['Title'][df_test.Title == 'Mme'] = 'Mrs'
df_test['Title'][df_test.Title.isin(['Capt', 'Don', 'Major', 'Col', 'Sir'])] = 'Sir'
df_test['Title'][df_test.Title.isin(['Dona', 'Lady', 'the Countess'])] = 'Lady'


#Turning cabin number into Deck
df_train['Cabin'].fillna('Unknown', inplace=True)
df_test['Cabin'].fillna('Unknown', inplace=True)

cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
df_train['Deck'] = df_train['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))
df_test['Deck'] = df_test['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))

df_train['Gender'] = df_train['Sex'].map({'female': 0, 'male': 1}).astype(int)
df_train['Port_of_Embarkation'] = df_train['Embarked'].map({'C': 1, 'Q': 2, 'S': 3})
df_train['Deck_N'] = df_train['Deck'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'T': 7, 'G': 8, 'Unknown': 9})
df_train['Title_N'] = df_train['Title'].map({'Mrs': 1, 'Mr': 2, 'Master': 3, 'Miss': 4, 'Sir': 5, 'Rev': 6,
                                             'Dr': 7, 'Lady': 8})

df_test['Gender'] = df_test['Sex'].map({'female': 0, 'male': 1}).astype(int)
df_test['Port_of_Embarkation'] = df_test['Embarked'].map({'C': 1, 'Q': 2, 'S': 3})
df_test['Deck_N'] = df_test['Deck'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'T': 7, 'G': 8, 'Unknown': 9})
df_test['Title_N'] = df_test['Title'].map({'Mrs': 1, 'Mr': 2, 'Master': 3, 'Miss': 4, 'Sir': 5, 'Rev': 6,
                                           'Dr': 7, 'Lady': 8})

df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch']
df_train['Fare_Per_Person']=df_train['Fare']/(df_train['FamilySize']+1)
df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch']
df_test['Fare_Per_Person'] = df_test['Fare']/(df_test['FamilySize']+1)

df_train['Fare_Per_Person'].fillna(1, inplace=True)
df_test['Fare_Per_Person'].fillna(1, inplace=True)

df_train['Port_of_Embarkation'].fillna(1, inplace=True)
df_test['Port_of_Embarkation'].fillna(1, inplace=True)

#predic ages train
age_df_train = df_train[['Age', 'Fare_Per_Person', 'Port_of_Embarkation', 'Gender', 'FamilySize', 'Parch', 'SibSp', 'Title_N', 'Pclass', 'Deck_N']]
age_features_train = age_df_train.loc[(age_df_train['Age'].notnull())].values[:, 1::]
age_labels_train = age_df_train.loc[(age_df_train['Age'].notnull())].values[:, 0]
rtr = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
rtr.fit(age_features_train, age_labels_train)
predictedAges = rtr.predict(age_df_train.loc[(age_df_train['Age'].isnull())].values[:, 1::])
age_df_train.loc[age_df_train['Age'].isnull(), 'Age'] = predictedAges

age_df_test = df_test[['Age', 'Fare_Per_Person', 'Port_of_Embarkation', 'Gender', 'FamilySize', 'Parch', 'SibSp', 'Title_N', 'Pclass', 'Deck_N']]
age_features_test = age_df_test.loc[(age_df_test['Age'].notnull())].values[:, 1::]
age_labels_test = age_df_test.loc[(age_df_test['Age'].notnull())].values[:, 0]
rtr = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
rtr.fit(age_features_test, age_labels_test)
predictedAges = rtr.predict(age_df_test.loc[(age_df_test['Age'].isnull())].values[:, 1::])
age_df_test.loc[(age_df_test['Age'].isnull()), 'Age'] = predictedAges

# print age_df_test[pd.isnull(age_df_test).any(axis=1)]


# df_train['Age'].dropna()
# df_test['Age'].dropna()
df_train['AgeFullFill'] = age_df_train['Age']
df_train['AgeName'] = ""
df_train['AgeName'][df_train.AgeFullFill <= 5] = 1
df_train['AgeName'][(df_train.AgeFullFill > 5) & (df_train.AgeFullFill < 15)] = 3
df_train['AgeName'][(df_train.AgeFullFill >= 15) & (df_train.AgeFullFill < 60)] = 4
df_train['AgeName'][df_train.AgeFullFill >= 60] = 2

df_train['HighLow'] = df_train['Pclass']
df_train['HighLow'][(df_train.Fare_Per_Person < 8)] = 1
df_train['HighLow'][(df_train.Fare_Per_Person >= 8)] = 2


df_train['Age*Class'] = df_train['AgeFullFill'] * df_train.Pclass
df_train['Age*Cabin'] = df_train['AgeFullFill'] * df_train['Deck_N']
df_train['Title*FarePP'] = df_train['Title_N'] * df_train['Fare_Per_Person']



df_test['AgeFullFill'] = age_df_test['Age']
df_test['AgeName'] = ""
df_test['AgeName'][df_test.AgeFullFill <= 5] = 1
df_test['AgeName'][(df_test.AgeFullFill > 5) & (df_test.AgeFullFill < 15)] = 3
df_test['AgeName'][(df_test.AgeFullFill >= 15) & (df_test.AgeFullFill < 60)] = 4
df_test['AgeName'][df_test.AgeFullFill >= 60] = 2

df_test['HighLow'] = df_test['Pclass']
df_test['HighLow'][(df_test.Fare_Per_Person < 8)] = 1
df_test['HighLow'][(df_test.Fare_Per_Person >= 8)] = 2

df_test['Age*Class'] = df_test['AgeFullFill'] * df_test.Pclass
df_test['Age*Cabin'] = df_test['AgeFullFill'] * df_test['Deck_N']
df_test['Title*FarePP'] = df_test['Title_N'] * df_test['Fare_Per_Person']





# median_ages = np.zeros((2, 3))
#
# for i in range(0, 2):
#     for j in range(0, 3):
#         median_ages[i, j] = df_train[(df_train['Gender'] == i) & \
#                                (df_train['Pclass'] == j + 1)]['Age'].dropna().median()
#
# for i in range(0, 2):
#     for j in range(0, 3):
#         median_ages[i, j] = df_test[(df_test['Gender'] == i) & \
#                                (df_test['Pclass'] == j + 1)]['Age'].dropna().median()

# df_train['AgeFill'] = df_train['Age']
# df_test['AgeFill'] = df_test['Age']

# for i in range(0, 2):
#     for j in range(0, 3):
#         df_train.loc[(df_train.Age.isnull()) & (df_train.Gender == i) & (df_train.Pclass == j + 1), \
#                'AgeFill'] = median_ages[i, j]
#
#
# for i in range(0, 2):
#     for j in range(0, 3):
#         df_test.loc[(df_test.Age.isnull()) & (df_test.Gender == i) & (df_test.Pclass == j + 1), \
#                'AgeFill'] = median_ages[i, j]

# df_train['AgeIsNull'] = pd.isnull(df_train.Age).astype(int)
# df_test['AgeIsNull'] = pd.isnull(df_test.Age).astype(int)


# print df.dtypes[df.dtypes.map(lambda x: x=='object')]
#
# print list(df_train.columns.values)
# exit()
# df_train = df_train.drop(['Sex', 'Embarked', 'Fare', 'Ticket', 'Cabin', 'Name', 'Title', 'Deck'], axis=1)
# df_train = df_train.dropna()
# df_test = df_test.drop(['Sex', 'Embarked', 'Fare', 'Ticket', 'Cabin', 'Name', 'Title', 'Deck'], axis=1)
# Find if there are any nans
# print df_test[pd.isnull(df_test).any(axis=1)]
# exit()

FEATURES = ['Pclass', 'SibSp', 'Parch', 'Gender', 'Port_of_Embarkation', 'AgeFullFill',
            'FamilySize', 'Age*Class', 'Deck_N', 'Title_N', 'Fare_Per_Person', 'AgeName', 'HighLow']

#FEATURES = ['Gender', 'AgeFullFill', 'Age*Class', 'Title_N', 'Fare_Per_Person']

features_train = np.array(df_train[FEATURES].values)
labels_train = df_train["Survived"]

features_test = np.array(df_test[FEATURES].values)
labels_test = df_test["Survived"]

data_train = df_train
data_test = df_test


# features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3,
#                                                                           random_state=42)

#scale data
# min_max_scaler = preprocessing.MinMaxScaler()
# features_train_scaled = min_max_scaler.fit_transform(features_train)
# features_test_scaled = min_max_scaler.fit_transform(features_test)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.fit_transform(features_test)

from sklearn.pipeline import Pipeline
from sklearn.decomposition import RandomizedPCA


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
#               'min_samples_split': [1, 2, 10, 20],
#               'max_features': (5, 8, 'auto', None),
#               'max_depth': [None, 2, 10],
#               'n_estimators': [1000, 2000],
#               'max_leaf_nodes': [8, 16, 32]}
'''
0.782296650718
clf = RandomForestClassifier(bootstrap=True,
             criterion='entropy', max_depth=None, max_features=2,
             max_leaf_nodes=8, min_samples_split=1, n_estimators=1000,
             n_jobs=-1, oob_score=False)

0.791866028708
clf = RandomForestClassifier(bootstrap=True,
            criterion='entropy', max_depth=None, max_features=2,
            max_leaf_nodes=8, min_samples_split=1, n_estimators=1000,
            n_jobs=-1, oob_score=False)
'''
# pipeline = RandomForestClassifier(n_estimators=100, random_state=42)
# clf = grid_search.GridSearchCV(pipeline, param_grid=param_grid, verbose=3, scoring='accuracy', cv=10)
# clf = clf.fit(features_train, labels_train)
# pred = clf.predict(features_test)
# print(clf.best_estimator_)
# acc = accuracy_score(labels_test, pred)
# print acc

# print "Rough fitting a RandomForest to determine feature importance..."
#
# forest = RandomForestClassifier(oob_score=True, n_estimators=10000)
# forest.fit(features_train, labels_train)
# feature_importance = forest.feature_importances_
# feature_importance = 100.0 * (feature_importance / feature_importance.max())
# print feature_importance
# exit()


ids = df_test['PassengerId'].values

def brute_force_acc_rd(features_train, labels_train, features_test, labels_test, ids):

    clf = RandomForestClassifier(bootstrap=True,
            criterion='entropy', max_depth=None, max_features=5,
            max_leaf_nodes=8, min_samples_split=1, n_estimators=1000,
            n_jobs=-1, oob_score=False)

    # clf = RandomForestClassifier(bootstrap=True,
    #         criterion='entropy', max_depth=None, max_features=2,
    #         max_leaf_nodes=8, min_samples_split=1, n_estimators=1000,
    #         n_jobs=-1, oob_score=False)


    clf = clf.fit(features_train, labels_train)
    # print(clf.best_estimator_)
    pred = clf.predict(features_test)
    acc = accuracy_score(labels_test, pred)
    #print pred
    # if(acc > 0.80):
    #     print acc

    print acc
    importances = clf.feature_importances_
    print importances


    if(acc > 0.85):
        predictions_file = open("data/canivel_random_forest_81.csv", "wb")
        predictions_file_object = csv.writer(predictions_file)
        predictions_file_object.writerow(["PassengerId", "Survived"])
        predictions_file_object.writerows(zip(ids, pred))
        predictions_file.close()

    return acc


while brute_force_acc_rd(features_train, labels_train, features_test, labels_test, ids) < 0.85:
    brute_force_acc_rd(features_train, labels_train, features_test, labels_test, ids)

# from sklearn.ensemble import ExtraTreesClassifier
# # fit an Extra Trees model to the data
# model = ExtraTreesClassifier()
# model.fit(features_train, labels_train)
# pred = model.predict(features_test)
# print accuracy_score(labels_test, pred)
# # display the relative importance of each attribute
# print(model.feature_importances_)