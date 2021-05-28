from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve,\
davies_bouldin_score as dbs, normalized_mutual_info_score as nmi
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
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from tqdm import tqdm
from sklearn.cluster import KMeans
from typing import Tuple
import pandas as pd
import itertools
import numpy as np
import argparse
import random
import umap
import sys

from CAC import specificity, sensitivity, best_threshold, predict_clusters, predict_clusters_cac,\
compute_euclidean_distance, calculate_gamma_old, calculate_gamma_new, cac, score

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ALL')
parser.add_argument('--init', default='RAND')
parser.add_argument('--classifier', default='ALL')
parser.add_argument('--verbose', default="False")
parser.add_argument('--decay', default='Fixed')
parser.add_argument('--alpha')
parser.add_argument('--k')
args = parser.parse_args()  

CLASSIFIER = args.classifier
INIT = args.init
if args.verbose == "True":
    VERBOSE = True
else:
    VERBOSE = False

datasets = ["adult", "cic", "creditcard", "diabetes",\
            "magic", "sepsis", "titanic"]

classifiers = ["LR", "SVM", "LDA", "Perceptron", "RF", "KNN", "SGD", "Ridge"]
# classifiers = ["RF", "KNN"]

def get_classifier(classifier):
    if classifier == "LR":
        model = LogisticRegression(random_state=0, max_iter=1000)
    elif classifier == "RF":
        model = RandomForestClassifier(n_estimators=10, random_state=0)
    elif classifier == "SVM":
        # model = SVC(kernel="linear", probability=True)
        model = LinearSVC(max_iter=10000)
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
        model = LogisticRegression(class_weight='balanced', random_state=0, max_iter=1000)
    return model

res = pd.DataFrame(columns=['Dataset', 'Classifier', 'alpha', 'k', \
    'Base_F1_mean', 'Base_AUC_mean',\
    'KM_F1_mean', 'KM_AUC_mean', \
    'CAC_F1_mean', 'CAC_AUC_mean'], dtype=object)

test_results = pd.DataFrame(columns=['Dataset', 'Classifier', 'alpha', 'k', \
    'Base_F1_mean', 'Base_AUC_mean',\
    'KM_F1_mean', 'KM_AUC_mean', \
    'CAC_F1_mean', 'CAC_AUC_mean'], dtype=object)
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

params = {
    "adult": [0.05,2],
    "cic": [0.05,2],
    "creditcard": [0.02,3],
    "diabetes": [2,2],
    "magic": [0.005,2],
    "sepsis": [0.015,5],
    "spambase": [1,2],
    "titanic" : [100, 2],
}

if args.dataset == "ALL":
    data = datasets
else:
    data = [args.dataset]

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

        i = 0
        scale = StandardScaler()

        if args.alpha is not None:
            alpha = float(args.alpha)
        else:
            alpha = params[DATASET][0]

        if args.k is not None:
            n_clusters = int(args.k)
        else:
            n_clusters = params[DATASET][1]

        print("Training Base classifier")
        n_splits = 5

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=108)
        base_scores = np.zeros((n_splits, 2))
        km_scores = np.zeros((n_splits, 2))
        cac_best_scores = np.zeros((n_splits, 2))
        cac_term_scores = np.zeros((n_splits, 2))

        for train, test in skf.split(X, y):
            clf = get_classifier(CLASSIFIER)
            # print("Iteration: " + str(i))
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
            X_train = scale.fit_transform(X_train)
            X_test = scale.fit_transform(X_test)
            clf.fit(X_train, y_train.ravel())
            preds = clf.predict(X_test)
            pred_proba = clf.predict_proba(X_test)
            # print("F1: ", f1_score(preds, y_test), "AUC:", roc_auc_score(y_test.ravel(), pred_proba[:,1]))
            base_scores[i][0] = f1_score(preds, y_test)
            base_scores[i][1] = roc_auc_score(y_test.ravel(), pred_proba[:,1])
            i += 1

        print(base_scores[:,0])
        print("\nTraining CAC")
        beta = -np.infty # do not change this
        alphas = [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1]
        if DATASET == "diabetes":
            alphas = [1, 2, 2.25, 2.5, 2.75, 3]
        elif DATASET == "creditcard" or DATASET == "adult":
            alphas = [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5]
        elif DATASET == "sepsis":
            alphas = [0.005, 0.008, 0.01, 0.015, 0.018, 0.02]
        elif DATASET == "cic":
            alphas = [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1]
        elif DATASET == "titanic":
            alphas = [0.01, 0.05, 0.08, 0.1, 0.2, 0.5, 0.8, 1]

        param_grid = {
        'alpha': alphas,
        # 'k': [2, 3, 4, 5]
        'k': [2]
        }

        X1, X_test, y1, y_test = train_test_split(X, y, stratify=y, random_state=108)

        best_alpha = 0
        best_score = 0
        test_f1_auc = [0, 0, 0, 0, 0, 0]
        keys, values = zip(*param_grid.items())
        combs = list(itertools.product(*values))
        # random.shuffle(combs)

        for v in combs:
            hyperparameters = dict(zip(keys, v)) 
            alpha = hyperparameters['alpha']
            n_clusters = params[DATASET][1] # Fix number of clusters to previous values
            print(hyperparameters)
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=108)

            km_scores = np.zeros((n_splits, 2))
            cac_best_scores = np.zeros((n_splits, 2))
            cac_term_scores = np.zeros((n_splits, 2))
            i = 0

            for train, val in skf.split(X1, y1):
                X_train, X_val, y_train, y_val = X1[train], X1[val], y1[train], y1[val]
                X_train = scale.fit_transform(X_train)
                X_val = scale.fit_transform(X_val)

                clustering = KMeans(n_clusters=n_clusters, random_state=0, max_iter=300)
                if INIT == "KM":
                    labels = clustering.fit(X_train).labels_
                elif INIT == "RAND":
                    labels = np.random.randint(0, n_clusters, [len(X_train)])

                cluster_centers, models, alt_labels, errors, seps, l1 = cac(X_train, labels, 100, np.ravel(y_train), alpha, beta, classifier=CLASSIFIER, verbose=VERBOSE)
                scores, loss = score(X_val, np.array(y_val), models, cluster_centers[1], alt_labels, alpha, classifier=CLASSIFIER, verbose=True)
                f1, auc = scores[1:3]
                idx = -1
                cac_term_scores[i, 0] = f1[idx]
                cac_term_scores[i, 1] = auc[idx]

                km_scores[i, 0] = f1[0]
                km_scores[i, 1] = auc[0]
                i += 1

            print("5-Fold Base scores", np.mean(base_scores, axis=0))
            print("5-Fold KMeans scores", np.mean(km_scores, axis=0))        
            print("5-Fold terminal CAC scores", np.mean(cac_term_scores, axis=0))
            print("\n")

            res.loc[res_idx] = [DATASET, CLASSIFIER, alpha, n_clusters] + list(np.mean(base_scores, axis=0)) + \
            list(np.mean(km_scores, axis=0)) + \
            list(np.mean(cac_term_scores, axis=0))
            res_idx += 1

            X1 = scale.fit_transform(X1)
            X_test = scale.fit_transform(X_test)

            if INIT == "KM":
                labels = clustering.fit(X1).labels_
            elif INIT == "RAND":
                labels = np.random.randint(0, n_clusters, [len(X1)])

            cluster_centers, models, alt_labels, errors, seps, l1 = cac(X1, labels, 100, np.ravel(y1), alpha, beta, classifier=CLASSIFIER, verbose=VERBOSE)
            scores, loss = score(X_val, np.array(y_val), models, cluster_centers[1], alt_labels, alpha, classifier=CLASSIFIER, verbose=True)
            f1, auc = scores[1:3]

            clf = get_classifier(CLASSIFIER)
            clf.fit(X1, y1.ravel())
            preds = clf.predict(X_test)
            pred_proba = clf.predict_proba(X_test)

            print("\nBase final test performance: ", "F1: ", f1_score(preds, y_test), "AUC: ", roc_auc_score(y_test.ravel(), pred_proba[:,1]))
            print("KM final test performance: ", "F1: ", f1[0], "AUC: ", auc[0])
            print("CAC final test performance: ", "F1: ", f1[-1], "AUC: ", auc[-1])
            print("\n")

            # Can choose whether it to do it w.r.t F1 or AUC
            if np.mean(cac_term_scores, axis=0)[1] > best_score:
                best_score = np.mean(cac_term_scores, axis=0)[1]
                best_alpha = alpha
                best_k = n_clusters
                test_f1_auc = [f1_score(preds, y_test), roc_auc_score(y_test.ravel(), pred_proba[:,1]), f1[0], auc[0], f1[-1], auc[-1]]
                # print(test_f1_auc)

        print(DATASET, ": Best alpha = ", best_alpha)
        test_results.loc[test_idx] = [DATASET, CLASSIFIER, best_alpha, best_k] + test_f1_auc
        print(test_results)
        test_idx += 1
    test_results.to_csv("./Results/Tuned_Test_Results" + ".csv", index=None)

res.to_csv("./Results/Tuning_every_run" + args.classifier + "_" + args.dataset + ".csv", index=None)
test_results.to_csv("./Results/Tuned_Test_Results" + args.classifier + "_" + args.dataset + ".csv", index=None)