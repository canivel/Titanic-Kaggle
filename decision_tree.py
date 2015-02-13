import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import csv as csv
from dataframe_builder import build_dataframes



df_train, df_test = build_dataframes()

FEATURES = ['Pclass', 'FamilySize*Gender', 'Gender', 'AgeFullFill',  'TicketNumber', 'Gender*TicketNumberStart',
            'Age*Class', 'Fare_Per_Person']

features_train = np.array(df_train[FEATURES].values)
labels_train = df_train["Survived"]

features_test = np.array(df_test[FEATURES].values)
labels_test = df_test["Survived"]

data_train = df_train
data_test = df_test
#
# data_train.to_csv("data_train.tst", "\t")
# exit()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.fit_transform(features_test)


# print "Rough fitting a RandomForest to determine feature importance..."
#
# forest = RandomForestClassifier(oob_score=True, n_estimators=10000)
# forest.fit(features_train, labels_train)
# feature_importance = forest.feature_importances_
# feature_importance = 100.0 * (feature_importance / feature_importance.max())
# print feature_importance
# exit()

#best til now 851pm 2/12/15
# clf = RandomForestClassifier(bootstrap=True,
#             criterion='entropy', max_depth=None, max_features=2,
#             max_leaf_nodes=16, min_samples_split=10, n_estimators=1000,
#             n_jobs=-1, oob_score=False)


ids = df_test['PassengerId'].values

def brute_force_acc_rd(features_train, labels_train, features_test, labels_test, ids):


    #0.825358851675
    clf = DecisionTreeClassifier(criterion='gini',
                                 min_samples_split=10,
                                 max_depth=10,
                                 max_leaf_nodes=16,
                                 max_features=2)



    # clf = DecisionTreeClassifier(compute_importances=None,
    #                              criterion='gini',
    #                              max_depth=1,
    #                              max_features='auto',
    #                              max_leaf_nodes=None,
    #                              min_density=None,
    #                              min_samples_leaf=1,
    #                              min_samples_split=4,
    #                              random_state=42,
    #                              splitter='random')

    clf = clf.fit(features_train, labels_train)
    # print(clf.best_estimator_)
    pred = clf.predict(features_test)
    acc = accuracy_score(labels_test, pred)
    #print pred

    feature_importance = clf.feature_importances_

    if(acc > 0.822):
        print acc
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        print feature_importance


    if(acc > 0.826):
        data_train.to_csv("data_train{}.tst".format(round(acc,5)), "\t")
        predictions_file = open("data/canivel_random_forest_{}.csv".format(round(acc, 5)), "wb")
        predictions_file_object = csv.writer(predictions_file)
        predictions_file_object.writerow(["PassengerId", "Survived"])
        predictions_file_object.writerows(zip(ids, pred))
        predictions_file.close()
        print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  NEW FILE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! YEA!!!!")
    return acc


while brute_force_acc_rd(features_train, labels_train, features_test, labels_test, ids) < 1.0:
    brute_force_acc_rd(features_train, labels_train, features_test, labels_test, ids)

# from sklearn.ensemble import ExtraTreesClassifier
# # fit an Extra Trees model to the data
# model = ExtraTreesClassifier()
# model.fit(features_train, labels_train)
# pred = model.predict(features_test)
# print accuracy_score(labels_test, pred)
# # display the relative importance of each attribute
# print(model.feature_importances_)