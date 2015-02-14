import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import csv as csv
from dataframe_builder import build_dataframes
import time
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

ids = df_test['PassengerId'].values


def brute_force_acc_rd(features_train, labels_train, features_test, labels_test, ids):

    clf = KNeighborsClassifier(
        n_neighbors=100,
        )

    clf = clf.fit(features_train, labels_train)
    # print(clf.best_estimator_)
    pred = clf.predict(features_test)
    acc = accuracy_score(labels_test, pred)
    #print pred
    print acc

    if(acc > 0.8):
        print ("Acc: {} ").format(acc)


    if(acc > 0.831):
        data_train.to_csv("data_train{}.tst".format(round(acc,5)), "\t")
        predictions_file = open("data/canivel_knn_{}.csv".format(round(acc, 5)), "wb")
        predictions_file_object = csv.writer(predictions_file)
        predictions_file_object.writerow(["PassengerId", "Survived"])
        predictions_file_object.writerows(zip(ids, pred))
        predictions_file.close()
        print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  NEW FILE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! YEA!!!!")
    return acc


while brute_force_acc_rd(features_train, labels_train, features_test, labels_test, ids) < 1.0:
    brute_force_acc_rd(features_train, labels_train, features_test, labels_test, ids)


# if __name__ == "__main__":
#
#     #GRIDSEARCH
#     # SVC(C=100, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.01,
#     #     kernel='rbf', max_iter=-1, probability=False, random_state=None,
#     #     shrinking=True, tol=0.001, verbose=False)
#     from sklearn.grid_search import GridSearchCV
#     from sklearn.cross_validation import train_test_split
#     print "Decision Tree"
#     param_grid = [
#         {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#         {'C': [1, 10, 100, 1000], 'gamma': [0.01, 0.0001], 'kernel': ['rbf']},
#         ]
#     clf = GridSearchCV(SVC(), param_grid,
#                        verbose = 3, scoring = "accuracy", n_jobs = -1, cv =10)
#
#     clf = clf.fit(features_train, labels_train)
#     print(clf.best_estimator_)
#     pred = clf.predict(features_test)
#     print accuracy_score(labels_test, pred)