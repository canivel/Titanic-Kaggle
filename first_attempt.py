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

df = pd.DataFrame.from_csv("data/train.csv")
df = df.drop(['Ticket', 'Cabin', 'Name'], axis=1)
df['Age'].dropna()

df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
df['Port_of_Embarkation'] = df['Embarked'].map({'C': 1, 'Q': 2, 'S': 3})

median_ages = np.zeros((2, 3))

for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i, j] = df[(df['Gender'] == i) & \
                               (df['Pclass'] == j + 1)]['Age'].dropna().median()

df['AgeFill'] = df['Age']

df[df['Age'].isnull()][['Gender', 'Pclass', 'Age', 'AgeFill']].head(10)

for i in range(0, 2):
    for j in range(0, 3):
        df.loc[(df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j + 1), \
               'AgeFill'] = median_ages[i, j]

df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
df['FamilySize'] = df['SibSp'] + df['Parch']
df['Age*Class'] = df.AgeFill * df.Pclass

# print df.dtypes[df.dtypes.map(lambda x: x=='object')]

df = df.drop(['Sex', 'Embarked', 'Age'], axis=1)
df = df.dropna()


FEATURES = ['Pclass', 'SibSp', 'Parch', 'Fare', 'Gender', 'Port_of_Embarkation', 'AgeFill', 'AgeIsNull',
            'FamilySize', 'Age*Class']

features = np.array(df[FEATURES].values)
labels = df["Survived"]

data = df

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3,
                                                                          random_state=42)

#scale data
min_max_scaler = preprocessing.MinMaxScaler()
features_train_scaled = min_max_scaler.fit_transform(features_train)
features_test_scaled = min_max_scaler.fit_transform(features_test)

forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(features_train, labels_train)
output = forest.predict(features_test)

print accuracy_score(labels_test, output)


def targetFeatureSplit( data ):

    target = []
    features = []
    for item in data:
        target.append( item[0] )
        features.append( item[1:] )

    return target, features