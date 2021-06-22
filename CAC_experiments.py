from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve,\
davies_bouldin_score as dbs, normalized_mutual_info_score as nmi, average_precision_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from scipy.stats import ttest_ind
from sklearn import model_selection, metrics
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from matplotlib import pyplot as plt
from datetime import datetime
from tqdm import tqdm
from sklearn.cluster import KMeans
from typing import Tuple
import pandas as pd
import numpy as np
import argparse
import umap
import sys

from CAC import CAC

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ALL')
parser.add_argument('--init', default='KM')
parser.add_argument('--classifier', default='ALL')
parser.add_argument('--verbose', default="False")
parser.add_argument('--decay', default='Fixed')
parser.add_argument('--cv', default='False')
parser.add_argument('--alpha')
parser.add_argument('--k')
args = parser.parse_args()

now = datetime.now()
time = now.strftime("%H:%M:%S")

CLASSIFIER = args.classifier
INIT = args.init
if args.verbose == "True":
    VERBOSE = True
else:
    VERBOSE = False

datasets = ["adult", "cic", "creditcard", "diabetes",\
            "magic", "titanic"]

classifiers = ["LR", "SVM", "LDA", "Perceptron", "RF", "KNN", "SGD", "Ridge"]

test_results = pd.DataFrame(columns=['Dataset', 'Classifier', 'alpha',\
    'Base_F1_mean', 'Base_AUC_mean', 'Base_F1_std', 'Base_AUC_std',\
    'KM_F1_mean', 'KM_AUC_mean', 'KM_F1_std', 'KM_AUC_std',\
    'CAC_F1_mean', 'CAC_AUC_mean', 'CAC_F1_std', 'CAC_AUC_std'], dtype=object)

def get_classifier(classifier):
    if classifier == "LR":
        model = LogisticRegression(random_state=0, max_iter=1000)
    elif classifier == "RF":
        model = RandomForestClassifier(n_estimators=10)
    elif classifier == "SVM":
        # model = SVC(kernel="linear", probability=True)
        model = LinearSVC(max_iter=5000)
        model.predict_proba = lambda X: np.array([model.decision_function(X), model.decision_function(X)]).transpose()
    elif classifier == "Perceptron":
        model = Perceptron()
        model.predict_proba = lambda X: np.array([model.decision_function(X), model.decision_function(X)]).transpose()
    elif classifier == "ADB":
        model = AdaBoostClassifier(n_estimators = 100)
    elif classifier == "DT":
        model = DecisionTreeClassifier()
    elif classifier == "LDA":
        model = LDA()
    elif classifier == "NB":
        model = MultinomialNB()
    elif classifier == "SGD":
        model = SGDClassifier(loss='log')
    elif classifier == "Ridge":
        model = RidgeClassifier()
        model.predict_proba = lambda X: np.array([model.decision_function(X), model.decision_function(X)]).transpose()
    elif classifier == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
    else:
        model = LogisticRegression(class_weight='balanced', max_iter=1000)
    return model

res = pd.DataFrame(columns=['Dataset', 'Classifier', 'alpha',\
    'Base_F1_mean', 'Base_AUC_mean', 'Base_F1_std', 'Base_AUC_std',\
    'KM_F1_mean', 'KM_AUC_mean', 'KM_F1_std', 'KM_AUC_std', 'KM-p-F1', 'KM-p-AUC',\
    'CAC_F1_mean', 'CAC_AUC_mean', 'CAC_F1_std', 'CAC_AUC_std', 'CAC-Base-p-F1', 'CAC-Base-p-AUC', 'CAC-KM-p-F1', 'CAC-KM-p-AUC'], dtype=object)

# alpha, #clusters
old_params = {
    "adult": [0.05,2],
    "cic": [0.1,2],
    "creditcard": [0.01,3],
    "diabetes": [2,2],
    "magic": [0.04,2],
    "sepsis": [0.015,5],
    "spambase": [1,2],
    "titanic": [100,2],
}

n_cluster_params = {
    "adult": 2,
    "cic": 2,
    "creditcard": 3,
    "diabetes": 2,
    "magic": 2,
    "sepsis": 5,
    "spambase": 2,
    "titanic": 2,
}

params = {'LR': {'titanic': 0.8,
  'magic': 0.01,
  'creditcard': 0.02,
  'adult': 0.05,
  'diabetes': 2.5,
  'sepsis': 0.005,
  'cic': 0.05},
 'SVM': {'titanic': 0.01,
  'magic': 0.02,
  'creditcard': 0.15,
  'adult': 0.15,
  'diabetes': 2.5,
  'sepsis': 0.008,
  'cic': 0.5},
 'LDA': {'titanic': 0.08,
  'magic': 0.05,
  'creditcard': 0.02,
  'adult': 0.15,
  'diabetes': 2.5,
  'sepsis': 0.05,
  'cic': 0.05},
 'Perceptron': {'titanic': 0.5,
  'magic': 0.01,
  'creditcard': 0.2,
  'adult': 0.15,
  'diabetes': 2.5,
  'sepsis': 0.005,
  'cic': 0.5},
 'RF': {'titanic': 0.05,
  'magic': 0.02,
  'creditcard': 0.15,
  'adult': 0.02,
  'diabetes': 2.5,
  'sepsis': 0.01,
  'cic': 0.5},
 'KNN': {'titanic': 0.01,
  'magic': 0.01,
  'creditcard': 0.15,
  'adult': 0.02,
  'diabetes': 2.5,
  'sepsis': 0.008,
  'cic': 0.5},
 'SGD': {'titanic': 0.3,
  'magic': 0.01,
  'creditcard': 0.08,
  'adult': 0.15,
  'diabetes': 2.5,
  'sepsis': 0.005,
  'cic': 0.01},
 'Ridge': {'titanic': 0.01,
  'magic': 0.02,
  'creditcard': 0.15,
  'adult': 0.15,
  'diabetes': 2.5,
  'sepsis': 0.015,
  'cic': 0.5}}

if args.dataset == "ALL":
    data = datasets
else:
    data = [args.dataset]

if args.classifier == "ALL":
    classifier = classifiers
else:
    classifier = [args.classifier]

res_idx = 0
test_idx = 0

for CLASSIFIER in classifier:
    for DATASET in data:
        print("Testing on Dataset: ", DATASET)
        print("Testing with Classifier: ", CLASSIFIER)
        ############ FOR CIC DATASET ############
        if DATASET == "cic":
            Xa = pd.read_csv("./data/CIC/cic_set_a.csv")
            Xb = pd.read_csv("./data/CIC/cic_set_b.csv")
            Xc = pd.read_csv("./data/CIC/cic_set_c.csv")

            ya = Xa['In-hospital_death']
            yb = Xb['In-hospital_death']
            yc = Xc['In-hospital_death']

            Xa = Xa.drop(columns=['recordid', 'Survival', 'In-hospital_death'])
            Xb = Xb.drop(columns=['recordid', 'Survival', 'In-hospital_death'])
            Xc = Xc.drop(columns=['recordid', 'Survival', 'In-hospital_death'])

            cols = Xa.columns

            scale = StandardScaler()
            Xa = scale.fit_transform(Xa)
            Xb = scale.fit_transform(Xb)
            Xc = scale.fit_transform(Xc)

            Xa = pd.DataFrame(Xa, columns=cols)
            Xb = pd.DataFrame(Xb, columns=cols)
            Xc = pd.DataFrame(Xc, columns=cols)

            Xa = Xa.fillna(0)
            Xb = Xb.fillna(0)
            Xc = Xc.fillna(0)

            X_train = pd.concat([Xa, Xb])
            y_train = pd.concat([ya, yb])

            X_test = Xc
            y_test = yc

            X = pd.concat([X_train, X_test]).to_numpy()
            y = pd.concat([y_train, y_test]).to_numpy()

        elif DATASET == "titanic":
            X_train = pd.read_csv("./data/" + DATASET + "/" + "X_train.csv").to_numpy()
            X_test = pd.read_csv("./data/" + DATASET + "/" + "X_test.csv").to_numpy()
            y_train = pd.read_csv("./data/" + DATASET + "/" + "y_train.csv").to_numpy()
            y_test = pd.read_csv("./data/" + DATASET + "/" + "y_test.csv").to_numpy()

            X = np.vstack([X_train, X_test])
            y = np.vstack([y_train, y_test])
            # X = pd.concat([X_train, X_test]).to_numpy()
            # y = pd.concat([y_train, y_test]).to_numpy()

        else:
            X = pd.read_csv("./data/" + DATASET + "/" + "X.csv").to_numpy()
            y = pd.read_csv("./data/" + DATASET + "/" + "y.csv").to_numpy()

        if args.alpha is not None:
            alpha = float(args.alpha)
        else:
            alpha = params[CLASSIFIER][DATASET]

        if args.k is not None:
            n_clusters = int(args.k)
        else:
            n_clusters = n_cluster_params[DATASET]

        beta = -np.infty # do not change this
        scale = StandardScaler()

        if args.cv == "True":
            n_splits = 5
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
            i = 0
            base_scores = np.zeros((n_splits, 2))
            km_scores = np.zeros((n_splits, 2))
            cac_best_scores = np.zeros((n_splits, 2))
            cac_term_scores = np.zeros((n_splits, 2))

            print("Training Base classifier")

            for train, test in skf.split(X, y):
                clf = get_classifier(CLASSIFIER)
                print("Iteration: " + str(i))
                X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
                X_train = scale.fit_transform(X_train)
                X_test = scale.fit_transform(X_test)
                clf.fit(X_train, y_train.ravel())
                preds = clf.predict(X_test)
                pred_proba = clf.predict_proba(X_test)
                print("F1: ", f1_score(preds, y_test), "AUC:", roc_auc_score(y_test.ravel(), pred_proba[:,1]))
                base_scores[i][0] = f1_score(preds, y_test)
                base_scores[i][1] = roc_auc_score(y_test.ravel(), pred_proba[:,1])
                i += 1

            print("\nTraining CAC")
            i = 0

            for train, test in skf.split(X, y):
                print("=============================")
                print("Stratified k-fold partition ", str(i))
                X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
                X_train = scale.fit_transform(X_train)
                X_test = scale.fit_transform(X_test)

                c = CAC(n_clusters, alpha, classifier=CLASSIFIER, verbose=VERBOSE)
                c.fit(X_train, y_train)
                # cluster_centers, models, alt_labels, errors, seps, l1 = cac(X_train, np.ravel(y_train), n_clusters, 100, alpha, beta, classifier=CLASSIFIER, verbose=VERBOSE)
                y_pred, y_proba = c.predict(X_test, -1)
                # scores, loss = score(X_test, np.array(y_test), models, cluster_centers[1], alt_labels, alpha, classifier=CLASSIFIER, verbose=True)
                f1, auc = f1_score(y_pred, y_test), roc_auc_score(y_test, y_proba)
                db = []
                for lbl_idx in range(len(c.labels)):
                    db.append(dbs(X_train, c.labels[lbl_idx]))

                if VERBOSE:
                    print("Best CAC Clustering")
                    idx = np.argmax(f1[1:])
                    cac_best_scores[i, 0] = f1[idx+1]
                    cac_best_scores[i, 1] = auc[idx+1]
                    # print(f1)
                    print("F1: ", f1[idx+1], "AUC: ", auc[idx+1], "DB: ", db[idx + 1], " at idx: ", idx+1)

                cac_term_scores[i, 0] = f1
                cac_term_scores[i, 1] = auc

                print("F1: ", f1, "AUC: ", auc, "DB: ", db[-1], " at idx: ", -1)

                y_pred, y_proba = c.predict(X_test, 0)
                f1, auc = f1_score(y_pred, y_test), roc_auc_score(y_test, y_proba)

                print("KMeans Clustering Score:")
                print("F1: ", f1, "AUC: ", auc, "DB: ", db[0])
                print("=============================\n")
                km_scores[i, 0] = f1
                km_scores[i, 1] = auc
                i += 1

            value, p0_f1 = ttest_ind(km_scores[:,0], base_scores[:,0])
            value, p0_auc = ttest_ind(km_scores[:,1], base_scores[:,1])

            value, p1_f1 = ttest_ind(cac_term_scores[:,0], base_scores[:,0])
            value, p1_auc = ttest_ind(cac_term_scores[:,1], base_scores[:,1])

            value, p2_f1 = ttest_ind(cac_term_scores[:,0], km_scores[:,0])
            value, p2_auc = ttest_ind(cac_term_scores[:,1], km_scores[:,1])


            print("5-Fold Base scores", np.mean(base_scores, axis=0))
            print("5-Fold KMeans scores", np.mean(km_scores, axis=0))        
            print("5-Fold best CAC scores", np.mean(cac_best_scores, axis=0))
            print("5-Fold terminal CAC scores", np.mean(cac_term_scores, axis=0), "p1 = ", [p1_f1, p1_auc], "p2 = ", [p2_f1, p2_auc])
            print("\n")

            res.loc[res_idx] = [DATASET, CLASSIFIER, alpha] + list(np.mean(base_scores, axis=0)) + list(np.std(base_scores, axis=0)) + \
            list(np.mean(km_scores, axis=0)) + list(np.std(km_scores, axis=0)) + [p0_f1, p0_auc] + \
            list(np.mean(cac_term_scores, axis=0)) + list(np.std(cac_term_scores, axis=0)) + [p1_f1, p1_auc] + [p2_f1, p2_auc]

            res_idx += 1
            res.to_csv("./Results/Results_5CV_" + time + ".csv")

        elif args.cv == "False":
            print("Testing CAC with heldout dataset 5 times")
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=108)
            X_train = scale.fit_transform(X_train)
            X_test = scale.fit_transform(X_test)

            epochs = 1
            base_scores = np.zeros((epochs, 3))
            km_scores = np.zeros((epochs, 3))
            cac_scores = np.zeros((epochs, 3))

            for i in range(epochs):
                c = CAC(n_clusters, alpha, classifier=CLASSIFIER, verbose=VERBOSE)
                c.fit(X_train, y_train)
                y_pred, y_proba = c.predict(X_test, -1)

                cac_scores[i, 0] = f1_score(y_pred, y_test)
                cac_scores[i, 1] = roc_auc_score(y_test, y_proba)
                cac_scores[i, 2] = average_precision_score(y_test, y_proba)
                print("CAC confusion matrix")
                print(confusion_matrix(y_test, y_pred))

                y_pred, y_proba = c.predict(X_test, 0)
                km_scores[i, 0] = f1_score(y_pred, y_test)
                km_scores[i, 1] = roc_auc_score(y_test, y_proba)
                km_scores[i, 2] = average_precision_score(y_test, y_proba)

                print("KM confusion matrix")
                print(confusion_matrix(y_test, y_pred))

                clf = get_classifier(CLASSIFIER)
                clf.fit(X_train, y_train.ravel())
                y_pred = clf.predict(X_test)
                pred_proba = clf.predict_proba(X_test)

                base_scores[i, 0] = f1_score(y_pred, y_test)
                base_scores[i, 1] = roc_auc_score(y_test.ravel(), pred_proba[:,1])
                base_scores[i, 2] = average_precision_score(y_test.ravel(), pred_proba[:,1])

                print("Base classifier confusion matrix")
                print(confusion_matrix(y_test, y_pred))

            print("Average Base scores", np.mean(base_scores, axis=0))
            print("Average KMeans scores", np.mean(km_scores, axis=0))        
            print("Average terminal CAC scores", np.mean(cac_scores, axis=0))
            print("\n")

            # test_results.loc[test_idx] = [DATASET, CLASSIFIER, alpha] + list(np.mean(base_scores, axis=0)) + list(np.std(base_scores, axis=0)) + \
            # list(np.mean(km_scores, axis=0)) + list(np.std(km_scores, axis=0)) + \
            # list(np.mean(cac_scores, axis=0)) + list(np.std(cac_scores, axis=0))
            # test_idx += 1
            # test_results.to_csv("./Results/Test_Results_STATIC_ALPHA_" + time + ".csv", index=None)