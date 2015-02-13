import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor
import string

df_train = pd.read_csv("data/train.csv", header=0)
df_test = pd.read_csv("data/test_acc.csv", header=0)

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
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col', 'Master']:
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

def getTicketNumber(ticket):
    match = re.compile("([\d]+$)").search(ticket)
    if match:
        return match.group()
    else:
        return '0'

def processTitle():
    title_list = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev', 'Dr', 'Ms', 'Mlle', 'Col', 'Capt', 'Mme', 'Countess', 'Don', 'Jonkheer']
    df_train['Title'] = df_train['Name'].map(lambda x: substrings_in_string(x, title_list))

    df_train['Title'].loc[df_train.Title == 'Jonkheer'] = 'Master'
    df_train['Title'].loc[df_train.Title.isin(['Ms', 'Mlle'])] = 'Miss'
    df_train['Title'].loc[df_train.Title == 'Mme'] = 'Mrs'
    df_train['Title'].loc[df_train.Title.isin(['Capt', 'Don', 'Major', 'Col', 'Sir', 'Rev', 'Dr'])] = 'Sir'
    df_train['Title'].loc[df_train.Title.isin(['Dona', 'Countess'])] = 'Lady'

    df_test['Title'] = df_test['Name'].map(lambda x: substrings_in_string(x, title_list))
    df_test['Title'].loc[df_test.Title == 'Jonkheer'] = 'Master'
    df_test['Title'].loc[df_test.Title.isin(['Ms', 'Mlle'])] = 'Miss'
    df_test['Title'].loc[df_test.Title == 'Mme'] = 'Mrs'
    df_test['Title'].loc[df_test.Title.isin(['Capt', 'Don', 'Major', 'Col', 'Sir', 'Rev', 'Dr'])] = 'Sir'
    df_test['Title'].loc[df_test.Title.isin(['Dona', 'Countess'])] = 'Lady'

    df_train['Title_N'] = df_train['Title'].map({'Mrs': 1, 'Mr': 2, 'Sir': 3, 'Miss': 4, 'Lady': 5, 'Master':6})
    df_test['Title_N'] = df_test['Title'].map({'Mrs': 1, 'Mr': 2, 'Sir': 3, 'Miss': 4, 'Lady': 5, 'Master':6})


def processCabin():
    #Turning cabin number into Deck
    df_train['Cabin'].fillna('Unknown', inplace=True)
    df_test['Cabin'].fillna('Unknown', inplace=True)

    cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
    df_train['Deck'] = df_train['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))
    df_test['Deck'] = df_test['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))
    df_train['Deck_N'] = df_train['Deck'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'T': 7, 'G': 8, 'Unknown': 9})
    df_test['Deck_N'] = df_test['Deck'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'T': 7, 'G': 8, 'Unknown': 9})

def processGender():
    df_train['Gender'] = df_train['Sex'].map({'female': 1, 'male': 2}).astype(int)
    df_test['Gender'] = df_test['Sex'].map({'female': 1, 'male': 2}).astype(int)

def processPortsOfEmbarkation():
    df_train['Port_of_Embarkation'] = df_train['Embarked'].map({'C': 1, 'Q': 2, 'S': 3})
    df_test['Port_of_Embarkation'] = df_test['Embarked'].map({'C': 1, 'Q': 2, 'S': 3})

    df_train['Port_of_Embarkation'].fillna(3, inplace=True)#S
    df_test['Port_of_Embarkation'].fillna(3, inplace=True)#S

def processFamily():
    df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch']
    df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch']

def processFare():
    df_train['Fare_Per_Person']=df_train['Fare']/(df_train['FamilySize']+1)
    df_test['Fare_Per_Person'] = df_test['Fare']/(df_test['FamilySize']+1)

    df_train['Fare_Per_Person'].fillna(32, inplace=True)#mean 32.2
    df_test['Fare_Per_Person'].fillna(32, inplace=True)#mean 32.2

    #EXTRAA FEATURES

    df_train['HighLow'] = df_train['Pclass']
    df_train['HighLow'].loc[(df_train.Fare_Per_Person < 52)] = 1 # people with fare 52 or less die more
    df_train['HighLow'].loc[(df_train.Fare_Per_Person >= 52)] = 2

    df_train['HighLow*Gender'] = df_train['Pclass']
    df_train['HighLow*Gender'].loc[(df_train.Fare_Per_Person < 52) & (df_train.Gender == 0)] = 1 # man poor
    df_train['HighLow*Gender'].loc[(df_train.Fare_Per_Person >= 52) & (df_train.Gender == 0)] = 2 #man rich
    df_train['HighLow*Gender'].loc[(df_train.Fare_Per_Person < 52) & (df_train.Gender == 1)] = 3 # woman poor
    df_train['HighLow*Gender'].loc[(df_train.Fare_Per_Person >= 52) & (df_train.Gender == 1)] = 4 #woman rich

    df_test['HighLow'] = df_test['Pclass']
    df_test['HighLow'].loc[(df_test.Fare_Per_Person < 52)] = 1 # people with fare 52 or less die more
    df_test['HighLow'].loc[(df_test.Fare_Per_Person >= 52)] = 2

    df_test['HighLow*Gender'] = df_test['Pclass']
    df_test['HighLow*Gender'].loc[(df_test.Fare_Per_Person < 52) & (df_test.Gender == 0)] = 1 # man poor
    df_test['HighLow*Gender'].loc[(df_test.Fare_Per_Person >= 52) & (df_test.Gender == 0)] = 2 #man rich
    df_test['HighLow*Gender'].loc[(df_test.Fare_Per_Person < 52) & (df_test.Gender == 1)] = 3 # woman poor
    df_test['HighLow*Gender'].loc[(df_test.Fare_Per_Person >= 52) & (df_test.Gender == 1)] = 4 #woman rich

    df_train['Title*FarePP'] = df_train['Title_N'] * df_train['Fare_Per_Person']
    df_test['Title*FarePP'] = df_test['Title_N'] * df_test['Fare_Per_Person']


def processTicket():

    df_train['TicketNumber'] = df_train['Ticket'].map(lambda x: getTicketNumber(x))
    df_train['TicketNumberDigits'] = df_train['TicketNumber'].map(lambda x: len(x)).astype(np.int)
    df_train['TicketNumberStart'] = df_train['TicketNumber'].map(lambda x: x[0:1]).astype(np.int)
    df_train['TicketNumber'] = df_train.TicketNumber.astype(np.int)

    df_test['TicketNumber'] = df_test['Ticket'].map(lambda x: getTicketNumber(x))
    df_test['TicketNumberDigits'] = df_test['TicketNumber'].map(lambda x: len(x)).astype(np.int)
    df_test['TicketNumberStart'] = df_test['TicketNumber'].map(lambda x: x[0:1]).astype(np.int)
    df_test['TicketNumber'] = df_test.TicketNumber.astype(np.int)

    #EXTRA FEATURES

    df_train['Port_of_Embarkation*TicketNumberStart'] = df_train['TicketNumberStart'] /  df_train['Port_of_Embarkation']
    df_test['Port_of_Embarkation*TicketNumberStart'] = df_test['TicketNumberStart'] / df_test['Port_of_Embarkation']

    df_train['Gender*TicketNumberStart'] = df_train['Gender'] / (df_train['TicketNumberStart']+1)
    df_test['Gender*TicketNumberStart'] = df_test['Gender'] / (df_test['TicketNumberStart']+1)

    df_train['FamilySize*TicketNumberStart'] = df_train['FamilySize'] / (df_train['TicketNumberStart']+1)
    df_test['FamilySize*TicketNumberStart'] = df_test['FamilySize'] / (df_test['TicketNumberStart']+1)

def processAge():
    #predic ages train
    age_df_train = df_train[['Age', 'Fare_Per_Person', 'Gender', 'FamilySize', 'Title_N', 'Pclass', 'Deck_N', 'TicketNumber', 'Gender*TicketNumberStart']]
    age_features_train = age_df_train.loc[(age_df_train['Age'].notnull())].values[:, 1::]
    age_labels_train = age_df_train.loc[(age_df_train['Age'].notnull())].values[:, 0]
    rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
    rtr.fit(age_features_train, age_labels_train)
    predictedAges = rtr.predict(age_df_train.loc[(age_df_train['Age'].isnull())].values[:, 1::])
    age_df_train.loc[age_df_train['Age'].isnull(), 'Age'] = predictedAges

    age_df_test = df_test[['Age', 'Fare_Per_Person', 'Gender', 'FamilySize', 'Title_N', 'Pclass', 'Deck_N', 'TicketNumber', 'Gender*TicketNumberStart']]
    age_features_test = age_df_test.loc[(age_df_test['Age'].notnull())].values[:, 1::]
    age_labels_test = age_df_test.loc[(age_df_test['Age'].notnull())].values[:, 0]
    rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
    rtr.fit(age_features_test, age_labels_test)
    predictedAges = rtr.predict(age_df_test.loc[(age_df_test['Age'].isnull())].values[:, 1::])
    age_df_test.loc[(age_df_test['Age'].isnull()), 'Age'] = predictedAges

    #AGE TYPES

    df_train['AgeFullFill'] = age_df_train['Age']
    df_train['AgeName'] = ""
    df_train['AgeName'].loc[df_train.AgeFullFill < 20] = 1
    df_train['AgeName'].loc[(df_train.AgeFullFill >= 20) & (df_train.AgeFullFill < 45)] = 2
    df_train['AgeName'].loc[df_train.AgeFullFill >= 45] = 3

    df_test['AgeFullFill'] = age_df_test['Age']
    df_test['AgeName'] = ""
    df_test['AgeName'].loc[df_test.AgeFullFill < 20] = 1
    df_test['AgeName'].loc[(df_test.AgeFullFill >= 20) & (df_test.AgeFullFill < 45)] = 2
    df_test['AgeName'].loc[df_test.AgeFullFill >= 45] = 3

    #EXTRA FEATURES

    df_train['AgeName*HighLow'] = ""
    df_train['AgeName*HighLow'] = df_train['AgeName']/df_train['HighLow']
    df_train['AgeName*HighLow'] = df_train['AgeName']/df_train['HighLow']

    df_train['AgeFullFill*HighLow'] = ""
    df_train['AgeFullFill*HighLow'] = df_train['HighLow']/df_train['AgeFullFill']
    df_train['AgeFullFill*HighLow'] = df_train['HighLow']/df_train['AgeFullFill']

    df_train['Age*Class'] = df_train['AgeFullFill'] * df_train.Pclass
    df_train['Age*Cabin'] = df_train['AgeFullFill'] * df_train['Deck_N']

    df_test['AgeName*HighLow'] = ""
    df_test['AgeName*HighLow'] = df_test['AgeName']/df_test['HighLow']
    df_test['AgeName*HighLow'] = df_test['AgeName']/df_test['HighLow']

    df_test['AgeFullFill*HighLow'] = ""
    df_test['AgeFullFill*HighLow'] = df_test['HighLow']/df_test['AgeFullFill']
    df_test['AgeFullFill*HighLow'] = df_test['HighLow']/df_test['AgeFullFill']

    df_test['Age*Class'] = df_test['AgeFullFill'] * df_test.Pclass
    df_test['Age*Cabin'] = df_test['AgeFullFill'] * df_test['Deck_N']


def build_dataframes():

    processTitle()
    processCabin()
    processGender()
    processPortsOfEmbarkation()
    processFamily()
    processFare()
    processTicket()
    processAge()


    return df_train, df_test


