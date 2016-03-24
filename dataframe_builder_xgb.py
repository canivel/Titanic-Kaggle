import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor
import string

df_train = pd.read_csv("data/train.csv", header=0, index_col=0)
df_test = pd.read_csv("data/test_acc.csv", header=0, index_col=0)

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

    df_train['FamilySize*Gender'] = df_train['FamilySize'] * df_train['Gender']
    df_test['FamilySize*Gender'] = df_test['FamilySize'] * df_test['Gender']



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

    df_train['FamilyType'] = ""
    df_train['FamilyType'].loc[(df_train['FamilySize'] > 2) & (df_train['HighLow'] == 1)] = 1
    df_train['FamilyType'].loc[(df_train['FamilySize'] == 2) & (df_train['HighLow'] == 1)] = 2
    df_train['FamilyType'].loc[(df_train['FamilySize'] <= 1) & (df_train['HighLow'] == 1)] = 3
    df_train['FamilyType'].loc[(df_train['FamilySize'] > 2) & (df_train['HighLow'] == 2)] = 4
    df_train['FamilyType'].loc[(df_train['FamilySize'] == 2) & (df_train['HighLow'] == 2)] = 5
    df_train['FamilyType'].loc[(df_train['FamilySize'] <= 1) & (df_train['HighLow'] == 2)] = 6

    df_test['FamilyType'] = ""
    df_test['FamilyType'].loc[(df_test['FamilySize'] > 2) & (df_test['HighLow'] == 1)] = 1
    df_test['FamilyType'].loc[(df_test['FamilySize'] == 2) & (df_test['HighLow'] == 1)] = 2
    df_test['FamilyType'].loc[(df_test['FamilySize'] <= 1) & (df_test['HighLow'] == 1)] = 3
    df_test['FamilyType'].loc[(df_test['FamilySize'] > 2) & (df_test['HighLow'] == 2)] = 4
    df_test['FamilyType'].loc[(df_test['FamilySize'] == 2) & (df_test['HighLow'] == 2)] = 5
    df_test['FamilyType'].loc[(df_test['FamilySize'] <= 1) & (df_test['HighLow'] == 2)] = 6


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
    age_df_train = df_train[['Age', 'Fare_Per_Person', 'Gender', 'FamilySize*Gender', 'FamilySize', 'Title_N',
                             'Pclass', 'Deck_N', 'TicketNumber', 'Gender*TicketNumberStart', 'SibSp',
                             'Parch', 'Port_of_Embarkation']]

    age_df_test = df_test[['Age', 'Fare_Per_Person', 'Gender', 'FamilySize*Gender', 'FamilySize', 'Title_N',
                           'Pclass', 'Deck_N', 'TicketNumber', 'Gender*TicketNumberStart', 'SibSp',
                           'Parch', 'Port_of_Embarkation']]

    age_features_train = age_df_train.loc[(age_df_train['Age'].notnull())].values[:, 1::]
    age_labels_train = age_df_train.loc[(age_df_train['Age'].notnull())].values[:, 0]

    # from sklearn import cross_validation
    # from sklearn.ensemble import GradientBoostingRegressor
    # from sklearn.metrics import mean_squared_error
    # features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(age_features_train, age_labels_train, test_size=0.3, random_state=42)
    #
    # # from sklearn.preprocessing import StandardScaler
    # # scaler = StandardScaler()
    # # features_train = scaler.fit_transform(features_train)
    # # features_test = scaler.fit_transform(features_test)
    #
    # clf = RandomForestRegressor(n_estimators=1000, n_jobs=-1,
    #                             max_features=5,
    #                             max_depth=None,
    #                             max_leaf_nodes=32,
    #                             min_samples_split=1,
    #                             min_samples_leaf =1)
    # clf = clf.fit(features_train, labels_train)
    # pred = clf.predict(features_test)
    #
    # #print clf.oob_score_
    # print 100.0 * (clf.feature_importances_ / clf.feature_importances_.max())
    # print clf.score(features_test, labels_test)
    # print mean_squared_error(labels_test, pred)
    #
    # exit()

    # from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from sklearn.ensemble import GradientBoostingRegressor

    rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
    #rtr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=20, max_features=2)
    rtr.fit(age_features_train, age_labels_train)
    predictedAges = rtr.predict(age_df_train.loc[(age_df_train['Age'].isnull())].values[:, 1::])
    age_df_train.loc[age_df_train['Age'].isnull(), 'Age'] = predictedAges

    age_features_test = age_df_test.loc[(age_df_test['Age'].notnull())].values[:, 1::]
    age_labels_test = age_df_test.loc[(age_df_test['Age'].notnull())].values[:, 0]
    rtr.fit(age_features_test, age_labels_test)
    predictedAges = rtr.predict(age_df_test.loc[(age_df_test['Age'].isnull())].values[:, 1::])
    age_df_test.loc[(age_df_test['Age'].isnull()), 'Age'] = predictedAges

    print 100.0 * (rtr.feature_importances_ / rtr.feature_importances_.max())

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

    df_train['AgeName*Gender'] = ""
    df_train['AgeName*Gender'].loc[(df_train['AgeName'] == 1) & (df_train['Gender'] == 1)] = 1
    df_train['AgeName*Gender'].loc[(df_train['AgeName'] == 2) & (df_train['Gender'] == 1)] = 2
    df_train['AgeName*Gender'].loc[(df_train['AgeName'] == 3) & (df_train['Gender'] == 1)] = 3
    df_train['AgeName*Gender'].loc[(df_train['AgeName'] == 1) & (df_train['Gender'] == 2)] = 4
    df_train['AgeName*Gender'].loc[(df_train['AgeName'] == 2) & (df_train['Gender'] == 2)] = 5
    df_train['AgeName*Gender'].loc[(df_train['AgeName'] == 3) & (df_train['Gender'] == 2)] = 6

    df_test['AgeName*Gender'] = ""
    df_test['AgeName*Gender'].loc[(df_test['AgeName'] == 1) & (df_test['Gender'] == 1)] = 1
    df_test['AgeName*Gender'].loc[(df_test['AgeName'] == 2) & (df_test['Gender'] == 1)] = 2
    df_test['AgeName*Gender'].loc[(df_test['AgeName'] == 3) & (df_test['Gender'] == 1)] = 3
    df_test['AgeName*Gender'].loc[(df_test['AgeName'] == 1) & (df_test['Gender'] == 2)] = 4
    df_test['AgeName*Gender'].loc[(df_test['AgeName'] == 2) & (df_test['Gender'] == 2)] = 5
    df_test['AgeName*Gender'].loc[(df_test['AgeName'] == 3) & (df_test['Gender'] == 2)] = 6


    df_train['AgeName*Gender*HighLow'] = ""
    df_train['AgeName*Gender*HighLow'].loc[(df_train['AgeName*Gender'] <= 3) & (df_train['HighLow'] == 1)] = 1 #poor
    df_train['AgeName*Gender*HighLow'].loc[(df_train['AgeName*Gender'] > 3) & (df_train['HighLow'] == 1)] = 2 #poor
    df_train['AgeName*Gender*HighLow'].loc[(df_train['AgeName*Gender'] <= 3) & (df_train['HighLow'] == 2)] = 1
    df_train['AgeName*Gender*HighLow'].loc[(df_train['AgeName*Gender'] > 3) & (df_train['HighLow'] == 2)] = 2

    df_test['AgeName*Gender*HighLow'] = ""
    df_test['AgeName*Gender*HighLow'].loc[(df_test['AgeName*Gender'] <= 3) & (df_test['HighLow'] == 1)] = 1
    df_test['AgeName*Gender*HighLow'].loc[(df_test['AgeName*Gender'] > 3) & (df_test['HighLow'] == 1)] = 2
    df_test['AgeName*Gender*HighLow'].loc[(df_test['AgeName*Gender'] <= 3) & (df_test['HighLow'] == 2)] = 1
    df_test['AgeName*Gender*HighLow'].loc[(df_test['AgeName*Gender'] > 3) & (df_test['HighLow'] == 2)] = 2

def processDrops():
    allFeaturesList = ['PassengerId', 'Pclass', 'Name', 'Sex',
                   'SibSp', 'Parch', 'Ticket', 'Fare','Cabin',
                   'Embarked','Title', 'Title_N', 'Deck', 'Deck_N',
                   'Gender', 'Port_of_Embarkation',	'FamilySize', 'Fare_Per_Person', 'HighLow',
                   'HighLow*Gender', 'Title*FarePP', 'TicketNumber', 'TicketNumberDigits', 'TicketNumberStart',
                   'Port_of_Embarkation*TicketNumberStart', 'Gender*TicketNumberStart', 'FamilySize*TicketNumberStart', 'AgeFullFill', 'AgeName',
                   'AgeName*HighLow', 'AgeFullFill*HighLow', 'Age*Class', 'Age*Cabin', 'FamilySize*Gender',
                   'AgeName*Gender', 'AgeName*Gender*HighLow', 'FamilyType', 'Survived']


    dropList = ['Name', 'Age', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Title', 'Deck']

    df_train.drop(dropList, axis=1, inplace=True)
    df_test.drop(dropList, axis=1, inplace=True)


def build_dataframes():

    processTitle()
    processCabin()
    processGender()
    processPortsOfEmbarkation()
    processFamily()
    processFare()
    processTicket()
    processAge()
    processDrops()

    return df_train, df_test


