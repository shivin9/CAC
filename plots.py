from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve,\
davies_bouldin_score as dbs, normalized_mutual_info_score as nmi, adjusted_rand_score as ars
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import log_loss
from sklearn import model_selection, metrics
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from matplotlib import pyplot as plt
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

from CAC import CAC


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ALL')
parser.add_argument('--init', default='RAND')
parser.add_argument('--classifier', default='LR')
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

DATASET = args.dataset
alphs = [0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 1, 1.5, 2, 3, 4]

def get_dataset(DATASET):
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

		y1 = []
		for i in range(len(y)):
			y1.append(y[i])
		y = np.array(y)
	return X, y


def compute_euclidean_distance(point, centroid):
    # return np.sum((point - centroid)**2)
    return np.sqrt(np.sum((point - centroid)**2))


def ics(X, y, k):
	X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=108)
	km = KMeans(n_clusters = k)
	km.fit(X_train)
	labels = km.labels_
	k = len(np.unique(labels))
	vals = np.zeros(k)
	lgloss = np.zeros(k) 
	mus = []
	N = len(X)
	base = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)
	base_loss = np.zeros(k)
	bounds_arr = []
	cluster_pts = np.zeros(k)
	local_models = []
	print("Base Loss: ", logloss(y_train, base.predict_proba(X_train)))
	for i in range(k):
		pts_index = np.where(labels == i)[0]
		model = LogisticRegression(random_state=0, max_iter=1000).fit(X_train[pts_index], y_train[pts_index])
		local_models.append(model)
		n_class_index = np.where(y_train[pts_index] == 0)[0]
		p_class_index = np.where(y_train[pts_index] == 1)[0]

		n_class = X_train[n_class_index]
		p_class = X_train[p_class_index]

		negative_centers = n_class.mean(axis=0)
		positive_centers = p_class.mean(axis=0)
		vals[i] = compute_euclidean_distance(positive_centers, negative_centers)
		lgloss[i] = logloss(y_train[pts_index], model.predict_proba(X_train[pts_index]))
		base_loss[i] = logloss(y_train[pts_index], base.predict_proba(X_train[pts_index]))
		cluster_pts[i] = len(pts_index)
		bounds_arr.append(bounds(X_train[pts_index], y_train[pts_index]))
		print("Cluster ", i, " ICS: ", vals[i])
		print("Cluster ", i, " local loss: ", lgloss[i])
		print("Cluster ", i, " global loss: ", base_loss[i])
		print("Cluster ", i, " Local AUC: ", roc_auc_score(y_train[pts_index], model.predict_proba(X_train[pts_index])[:,1]))
		print("Cluster ", i, " Global AUC: ", roc_auc_score(y_train[pts_index], base.predict_proba(X_train[pts_index])[:,1]))

		print("Cluster ", i, " Local ACC: ", accuracy_score(y_train[pts_index], model.predict(X_train[pts_index])))
		print("Cluster ", i, " Global ACC: ", accuracy_score(y_train[pts_index], base.predict(X_train[pts_index])))

		print("Cluster ", i, " Bounds: ", bounds_arr[-1])
		print("===============\n")
		# logloss[i] = log_loss(y_train[pts_index],  np.max(model.predict_proba(X_train[pts_index]), axis=1))
		# base_loss[i] = log_loss(y_train[pts_index],  np.max(base.predict_proba(X_train[pts_index]), axis=1))
		mu = np.mean(X_train[pts_index], axis=0)
		mus.append(mu)

	# perimeter = 0
	# if k > 1:
	# 	for i in range(len(mus)):
	# 		perimeter += compute_euclidean_distance(mus[i%k], mus[(i+1)%k])
	print("Total Training Local Loss: ", np.sum(lgloss))
	print("Total Training Global Loss: ", np.sum(base_loss))

	print("##########################")

	print("Base Testing Loss: ", logloss(y_test, base.predict_proba(X_test)))
	labels = km.predict(X_test)
	for i in range(k):
		pts_index = np.where(labels == i)[0]

		n_class_index = np.where(y_test[pts_index] == 0)[0]
		p_class_index = np.where(y_test[pts_index] == 1)[0]

		n_class = X_test[n_class_index]
		p_class = X_test[p_class_index]
		model = local_models[i]

		negative_centers = n_class.mean(axis=0)
		positive_centers = p_class.mean(axis=0)
		vals[i] = compute_euclidean_distance(positive_centers, negative_centers)
		lgloss[i] = logloss(y_test[pts_index], model.predict_proba(X_test[pts_index]))
		base_loss[i] = logloss(y_test[pts_index], base.predict_proba(X_test[pts_index]))
		cluster_pts[i] = len(pts_index)
		bounds_arr.append(bounds(X_test[pts_index], y_test[pts_index]))
		print("Cluster ", i, " ICS: ", vals[i])
		print("Cluster ", i, " local loss: ", lgloss[i])
		print("Cluster ", i, " global loss: ", base_loss[i])
		print("Cluster ", i, " Bounds: ", bounds_arr[-1])
		print("Cluster ", i, " Local AUC: ", roc_auc_score(y_test[pts_index], model.predict_proba(X_test[pts_index])[:,1]))
		print("Cluster ", i, " Global AUC: ", roc_auc_score(y_test[pts_index], base.predict_proba(X_test[pts_index])[:,1]))

		print("Cluster ", i, " Local ACC: ", accuracy_score(y_test[pts_index], model.predict(X_test[pts_index])))
		print("Cluster ", i, " Global ACC: ", accuracy_score(y_test[pts_index], base.predict(X_test[pts_index])))

		print("===============\n")
		# logloss[i] = log_loss(y_test[pts_index],  np.max(model.predict_proba(X_test[pts_index]), axis=1))
		# base_loss[i] = log_loss(y_test[pts_index],  np.max(base.predict_proba(X_test[pts_index]), axis=1))
		mu = np.mean(X_test[pts_index], axis=0)
		mus.append(mu)

	# perimeter = 0
	# if k > 1:
	# 	for i in range(len(mus)):
	# 		perimeter += compute_euclidean_distance(mus[i%k], mus[(i+1)%k])
	print("Total Testing Local Loss: ", np.sum(lgloss))
	print("Total Testing Global Loss: ", np.sum(base_loss))
	return None


def plot(DATASET, n_clusters=2):
	test_loss = []
	train_loss = []
	clustering_scores = []
	X, y = get_dataset(DATASET)
	scale = StandardScaler()
	X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=108)
	X_train = scale.fit_transform(X_train)
	X_test = scale.fit_transform(X_test)
	pts = []
	for alpha in alphs:
		c = CAC(n_clusters, alpha, verbose=True)
		c.fit(X_train, y_train)
		train_loss.append(c.classification_loss)
		clustering_scores.append(c.clustering_loss)
		# print(c.clustering_loss)
		pts.append(c.cluster_stats)
		# scores, loss = score(X_test, np.array(y_test), models, cluster_centers[1], alt_labels, alpha, classifier=CLASSIFIER, flag="old", verbose=True)
		# print("Test loss: ", loss)
		# test_loss.append(loss)
		# test_scores.append(scores[1][-1])
		# print("KM avg. sep", np.sum(c.clustering_loss[0][:,1]/np.sum(c.cluster_stats[0], axis=1)))
		# print("CAC avg. sep", np.sum(c.clustering_loss[-1][:,1]/np.sum(c.cluster_stats[-1], axis=1)))

		# print("KM avg. km", np.sum(c.clustering_loss[0][:,0]/np.sum(c.cluster_stats[0], axis=1)))
		# print("CAC avg. km", np.sum(c.clustering_loss[-1][:,0]/np.sum(c.cluster_stats[-1], axis=1)))

	return train_loss, clustering_scores, pts

'''
lines = []
for c in classifiers: 
     base, km, cac = "", "", "" 
     base += c + " & " 
     km   += "$KM$ + " + c + " & " 
     cac  += "CAC + "  + c + " & " 
     for d in datasets: 
         row = res[(res['Dataset'] == d) & (res['Classifier'] == c)] 
         base += "{$" + str(row.Base_F1_mean.values[0]) + "$}" + " & " 
         km   += "{$" + str(row.KM_F1_mean.values[0]) + "$}" + " & " 
         cac  += "{$" + str(row.CAC_F1_mean.values[0]) + "$}" + " & " 
     lines.append([base, km, cac])
'''