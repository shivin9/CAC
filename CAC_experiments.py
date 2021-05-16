from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve,\
davies_bouldin_score as dbs, normalized_mutual_info_score as nmi
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from scipy.stats import ttest_ind
from sklearn import model_selection, metrics
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.cluster import KMeans
from typing import Tuple
import pandas as pd
import numpy as np
import argparse
import umap
import sys

from CAC import specificity, sensitivity, best_threshold, predict_clusters, predict_clusters_cac,\
compute_euclidean_distance, calculate_gamma_old, calculate_gamma_new,\
cac, get_new_accuracy, score

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ALL')
parser.add_argument('--init', default='RAND')
parser.add_argument('--classifier', default='LR')
parser.add_argument('--alpha', default=0.04)
args = parser.parse_args()  

CLASSIFIER = args.classifier
INIT = args.init

datasets = ["adult", "cic", "creditcard", "diabetes",\
            "magic", "sepsis", "titanic"]

def get_classifier(classifier):
    if classifier == "LR":
        model = LogisticRegression(class_weight='balanced', random_state=0, max_iter=1000)
    elif classifier == "RF":
        model = RandomForestClassifier(n_estimators=10, random_state=0)
    elif classifier == "SVM":
        model = SVC(kernel="linear", probability=True)
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
    elif classifier == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
    else:
        model = LogisticRegression(class_weight='balanced', random_state=0, max_iter=1000)
    return model

res = pd.DataFrame(columns=['Dataset', 'Classifier', \
    'Base_F1_mean', 'Base_AUC_mean', 'Base_F1_std', 'Base_AUC_std',\
    'KM_F1_mean', 'KM_AUC_mean', 'KM_F1_std', 'KM_AUC_std', 'KM-p-F1', 'KM-p-AUC',\
    'CAC_F1_mean', 'CAC_AUC_mean', 'CAC_F1_std', 'CAC_AUC_std', 'CAC-Base-p-F1', 'CAC-Base-p-AUC', 'CAC-KM-p-F1', 'CAC-KM-p-AUC'])

# alpha, #clusters
params = {
    "adult": [0.1,2],
    "cic": [0.04,2],
    "creditcard": [0.04,3],
    "diabetes": [2,2],
    "magic": [0.04,2],
    "sepsis": [0.015,5],
    "spambase": [1,2],
    "titanic": [2,2],
}

if args.dataset == "ALL":
    data = datasets
else:
    data = [args.dataset]

res_idx = 0

for DATASET in data:
    print("Testing on Dataset: ", DATASET)

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

        X_train = pd.concat([Xa, Xb]).to_numpy()
        y_train = pd.concat([ya, yb]).to_numpy()

        X_test = Xc.to_numpy()
        y_test = yc.to_numpy()

    elif DATASET == "titanic":
        X_train = pd.read_csv("./data/" + DATASET + "/" + "X_train.csv").to_numpy()
        X_test = pd.read_csv("./data/" + DATASET + "/" + "X_test.csv").to_numpy()
        y_train = pd.read_csv("./data/" + DATASET + "/" + "y_train.csv").to_numpy()
        y_test = pd.read_csv("./data/" + DATASET + "/" + "y_test.csv").to_numpy()

    else:
        X = pd.read_csv("./data/" + DATASET + "/" + "X.csv").to_numpy()
        y = pd.read_csv("./data/" + DATASET + "/" + "y.csv").to_numpy()

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    i = 0
    scale = StandardScaler()
    base_scores = np.zeros((n_splits, 2))
    km_scores = np.zeros((n_splits, 2))
    cac_best_scores = np.zeros((n_splits, 2))
    cac_term_scores = np.zeros((n_splits, 2))
    km_scores = np.zeros((n_splits, 2))
    alpha = params[DATASET][0]
    n_clusters = params[DATASET][1]


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
        beta = -np.infty # do not change this
        clustering = KMeans(n_clusters=n_clusters, random_state=0, max_iter=300)
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        X_train = scale.fit_transform(X_train)
        X_test = scale.fit_transform(X_test)

        if INIT == "KM":
            labels = clustering.fit(X_train).labels_
        elif INIT == "RAND":
            labels = np.random.randint(0, n_clusters, [len(X_train)])

        cluster_centers, models, alt_labels, errors, seps, l1 = cac(X_train, labels, 100, np.ravel(y_train), alpha, beta, classifier=CLASSIFIER, verbose=True)
        # print("nmi: ", nmi(labels_km, alt_labels[-1]))
        f1, auc = score(X_test, np.array(y_test), models, cluster_centers[1], alt_labels, alpha, flag="old", verbose=True)[1:3]
        db = []
        for k in range(len(alt_labels)):
            db.append(dbs(X_train, alt_labels[k]))

        # print("Best CAC Clustering")
        # idx = np.argmax(f1[1:])
        # cac_best_scores[i, 0] = f1[idx+1]
        # cac_best_scores[i, 1] = auc[idx+1]
        # print("F1: ", f1[idx+1], "AUC: ", auc[idx+1], "DB: ", db[idx + 1], " at idx: ", idx+1)

        idx = len(f1) - 1
        cac_term_scores[i, 0] = f1[idx]
        cac_term_scores[i, 1] = auc[idx]

        print("F1: ", f1[idx], "AUC: ", auc[idx], "DB: ", db[idx], " at idx: ", idx)
        # print(f1, auc, db)
        # KMeans clustering models
        labels = clustering.fit(X_train).labels_
        cluster_centers, models, alt_labels, errors, seps, l1 = cac(X_train, labels, 0, np.ravel(y_train), alpha, beta, classifier=CLASSIFIER, verbose=True)
        f1, auc = score(X_test, np.array(y_test), models, cluster_centers[1], alt_labels, alpha, flag="old", verbose=True)[1:3]

        print("KMeans Clustering Score:")
        print("F1: ", f1[0], "AUC: ", auc[0], "DB: ", dbs(X_train, alt_labels[-1]))
        print("=============================\n")
        km_scores[i, 0] = f1[0]
        km_scores[i, 1] = auc[0]
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
    print(res)
    res.loc[res_idx] = [DATASET, CLASSIFIER] + list(np.mean(base_scores, axis=0)) + list(np.std(base_scores, axis=0)) + \
    list(np.mean(km_scores, axis=0)) + list(np.std(km_scores, axis=0)) + [p0_f1, p0_auc] + \
    list(np.mean(cac_term_scores, axis=0)) + list(np.std(cac_term_scores, axis=0)) + [p1_f1, p1_auc] + [p2_f1, p2_auc]
    res_idx += 1
    res.to_csv("Results.csv")