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
from sklearn.ensemble import RandomForestRegressor

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

df_train['Title'].loc[df_train.Title == 'Jonkheer'] = 'Master'
df_train['Title'].loc[df_train.Title.isin(['Ms', 'Mlle'])] = 'Miss'
df_train['Title'].loc[df_train.Title == 'Mme'] = 'Mrs'
df_train['Title'].loc[df_train.Title.isin(['Capt', 'Don', 'Major', 'Col', 'Sir'])] = 'Sir'
df_train['Title'].loc[df_train.Title.isin(['Dona', 'Lady', 'the Countess'])] = 'Lady'

df_test['Title'] = df_test['Name'].map(lambda x: substrings_in_string(x, title_list))
df_test['Title'] = df_test.apply(replace_titles, axis=1)

df_test['Title'].loc[df_test.Title == 'Jonkheer'] = 'Master'
df_test['Title'].loc[df_test.Title.isin(['Ms', 'Mlle'])] = 'Miss'
df_test['Title'].loc[df_test.Title == 'Mme'] = 'Mrs'
df_test['Title'].loc[df_test.Title.isin(['Capt', 'Don', 'Major', 'Col', 'Sir'])] = 'Sir'
df_test['Title'].loc[df_test.Title.isin(['Dona', 'Lady', 'the Countess'])] = 'Lady'

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
df_train['AgeName'].loc[df_train.AgeFullFill <= 5] = 1
df_train['AgeName'].loc[(df_train.AgeFullFill > 5) & (df_train.AgeFullFill < 15)] = 3
df_train['AgeName'].loc[(df_train.AgeFullFill >= 15) & (df_train.AgeFullFill < 60)] = 4
df_train['AgeName'].loc[df_train.AgeFullFill >= 60] = 2

df_train['HighLow'] = df_train['Pclass']
df_train['HighLow'].loc[(df_train.Fare_Per_Person < 8)] = 1
df_train['HighLow'].loc[(df_train.Fare_Per_Person >= 8)] = 2

df_train['Age*Class'] = df_train['AgeFullFill'] * df_train.Pclass
df_train['Age*Cabin'] = df_train['AgeFullFill'] * df_train['Deck_N']
df_train['Title*FarePP'] = df_train['Title_N'] * df_train['Fare_Per_Person']

df_test['AgeFullFill'] = age_df_test['Age']
df_test['AgeName'] = ""
df_test['AgeName'].loc[df_test.AgeFullFill <= 5] = 1
df_test['AgeName'].loc[(df_test.AgeFullFill > 5) & (df_test.AgeFullFill < 15)] = 3
df_test['AgeName'].loc[(df_test.AgeFullFill >= 15) & (df_test.AgeFullFill < 60)] = 4
df_test['AgeName'].loc[df_test.AgeFullFill >= 60] = 2

df_test['HighLow'] = df_test['Pclass']
df_test['HighLow'].loc[(df_test.Fare_Per_Person < 8)] = 1
df_test['HighLow'].loc[(df_test.Fare_Per_Person >= 8)] = 2

df_test['Age*Class'] = df_test['AgeFullFill'] * df_test.Pclass
df_test['Age*Cabin'] = df_test['AgeFullFill'] * df_test['Deck_N']
df_test['Title*FarePP'] = df_test['Title_N'] * df_test['Fare_Per_Person']


FEATURES = ['Pclass', 'SibSp', 'Parch', 'Gender', 'Port_of_Embarkation', 'AgeFullFill',
            'FamilySize', 'Age*Class', 'Deck_N', 'Title_N', 'Fare_Per_Person', 'AgeName', 'HighLow']

#FEATURES = ['Gender', 'AgeFullFill', 'Age*Class', 'Title_N', 'Fare_Per_Person']

features_train = np.array(df_train[FEATURES].values)
labels_train = df_train["Survived"]

features_test = np.array(df_test[FEATURES].values)
labels_test = df_test["Survived"]

data_train = df_train
data_test = df_test


from sklearn.preprocessing import StandardScaler, MinMaxScaler
#scale data
min_max_scaler = MinMaxScaler()
features_train_scaled = min_max_scaler.fit_transform(features_train)
features_test_scaled = min_max_scaler.fit_transform(features_test)

# scaler = StandardScaler()
# features_train_scaled = scaler.fit_transform(features_train)
# features_test_scaled = scaler.fit_transform(features_test)

from sklearn.metrics import accuracy_score
from sklearn import svm, grid_search

# param_grid = {'kernel': ('rbf', 'sigmoid'),
#               'C': [1, 10, 100]}
#
# clf = grid_search.GridSearchCV(svm.SVC(), param_grid, verbose=3)
clf = svm.SVC(C=1, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
              kernel='rbf', max_iter=-1, probability=False, random_state=None,
              shrinking=True, tol=0.001, verbose=False)
clf = clf.fit(features_train_scaled, labels_train)
# print(clf.best_estimator_)
pred = clf.predict(features_test_scaled)
# print pred
print accuracy_score(labels_test, pred)

