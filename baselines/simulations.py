from CAC import specificity, sensitivity, best_threshold, predict_clusters, predict_clusters_cac,\
predict_clusters_knn, compute_euclidean_distance, calculate_gamma_old, calculate_gamma_new,\
cac, get_new_accuracy, score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn import model_selection, metrics
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.cluster import KMeans
from typing import Tuple
import pandas as pd
import numpy as np
import sys
import umap
import argparse

color = ['grey', 'red', 'blue', 'pink', 'brown', 'black', 'magenta', 'purple', 'orange', 'cyan', 'olive']
cols = ["ICS", "OCS", "K", "k", "Trn_F1", "Trn_AUC",  "Tst_F1", "Tst_AUC"]

df = pd.DataFrame(columns=cols)
# df = pd.read_csv("simres.csv", delimiter='\t')
cnt = 0
for ics in ICS:
    for ocs in OCS:
        for data_k in K:
            for algo_k in k:
                key = str(ics) + "," + str(ocs) + "," + str(data_k) + "," + str(algo_k)
                if key in res:
                    # df.loc[cnt] = [ics, ocs, data_k, algo_k] + list(np.mean(res[key],axis=1).ravel())
                    df.loc[cnt] = [ics, ocs, data_k, algo_k] + list(np.hstack([res[key][0][1], res[key][1][1]]))
                    cnt += 1

def plot(df, grp_by):
	plt.plot(df.groupby(by=grp_by).mean()['Tst_F1'], label='Test_F1')
	plt.plot(df.groupby(by=grp_by).mean()['Tst_AUC'], label='Test_AUC')
	plt.xlabel(grp_by[0])
	plt.legend()
	plt.show()

# plt.plot(df.groupby(by=['K', 'k']).mean()	[['Tst_AUC']].loc[(3,)], label='AUC_K3')
# plt.plot(df.groupby(by=['K', 'k']).mean()[['Tst_F1']].loc[(3,)], label='F1_K3')
# plt.xlabel('k')
# plt.legend()
# plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('--n_features', type=int, default=8)
parser.add_argument('--n_samples', type=int, default=500)
parser.add_argument('--n_classes', type=int, default=2)
parser.add_argument("--n_informative", type=int, default=5)
parser.add_argument("--n_clusters", type=int, default=4)
parser.add_argument("--n_classifiers", type=int, default=4)
parser.add_argument("--n_clusterings", type=int, default=5)
parser.add_argument("--n_reduced_features", type=int, default=4)
parser.add_argument("--inner_class_sep", type=float, default=0.8)
parser.add_argument("--outer_class_sep", type=float, default=1.2)
parser.add_argument("--alpha", type=float, default=4)

args = parser.parse_args()

params = {}
n_features = args.n_features
n_samples = args.n_samples
n_classes = args.n_classes
n_informative = args.n_informative
n_clusters = args.n_clusters
n_classifiers = args.n_classifiers
n_clusterings = args.n_clusterings
n_reduced_features = args.n_reduced_features
outer_class_sep = args.outer_class_sep
inner_class_sep = args.inner_class_sep
alpha = args.alpha


params["n_features"] = n_features
params["n_samples"] = n_samples
params["n_classes"] = n_classes
params["n_informative"] = n_informative
params["n_clusters"] = n_clusters
params["n_classifiers"] = n_classifiers
params["n_clusterings"] = n_clusterings
params["n_reduced_features"] = n_reduced_features
params["outer_class_sep"] = outer_class_sep
params["inner_class_sep"] = inner_class_sep
params["alpha"] = alpha


def create_imbalanced_data_clusters(n_samples=n_samples, n_features=n_features, n_informative=n_informative, n_classes=n_classes,\
							n_clusters = 2, frac=0.2, outer_class_sep=0.15, inner_class_sep=0.1, clus_per_class=2):
	X = np.empty(shape=n_features)
	Y = np.empty(shape=1)
	offsets = np.random.normal(0, outer_class_sep, size=(n_clusters, n_features))
	for i in range(n_clusters):
		samples = int(np.random.normal(n_samples, n_samples/10))
		x, y = make_classification(n_samples=samples, n_features=n_features, n_informative=n_informative,\
									n_classes=n_classes, class_sep=inner_class_sep, n_clusters_per_class=clus_per_class)
									# n_repeated=0, n_redundant=0)
		x += offsets[i]
		y_0 = np.where(y == 0)[0]
		y_1 = np.where(y != 0)[0]
		y_1 = np.random.choice(y_1, int(np.random.normal(frac, frac/4)*len(y_1)))
		index = np.hstack([y_0,y_1])
		np.random.shuffle(index)
		x_new = x[index]
		y_new = y[index]

		X = np.vstack((X,x_new))
		Y = np.hstack((Y,y_new))

	X = X[1:,:]
	Y = Y[1:]
	return X, Y


print(params)

reducer1 = umap.UMAP(n_components=n_reduced_features)
reducer = umap.UMAP()

n_iter = 10
ICS = [0, 0.2, 0.5, 1, 1.5, 2]
OCS = [0, 0.5, 1, 1.5, 2]
# OCS = [0]
K = [2, 3, 5, 10, 15, 20, 30]
k = [2, 3, 4, 5]

scale = StandardScaler()
results = {}
flag = 1
alpha = 2
for ics in [ICS[4]]:
	# if flag == 1:
	# 	temp_ocs = OCS[1:]
	# else:
	# 	temp_ocs = OCS
	for ocs in OCS:
		for data_k in K[5:]:
			X_orig, y = create_imbalanced_data_clusters(n_samples=n_samples, n_features=n_features, n_informative=n_informative,\
														n_classes=n_classes, frac=0.2, clus_per_class=2, outer_class_sep=ocs,\
														inner_class_sep=ics, n_clusters=data_k)
			for algo_k in k:
				key = str(ics) + "," + str(ocs) + "," + str(data_k) + "," + str(algo_k)
				print(key)
				X = X_orig
				scores = []
				X = scale.fit_transform(X)
				y = np.array(y)
				skf = StratifiedKFold(n_splits=5, shuffle=True)
				cnt = 0
				clustering = KMeans(n_clusters=algo_k, random_state=0, max_iter=300)
				train_stats = np.zeros((n_iter+1, 2))
				test_stats  = np.zeros((n_iter+1, 2))

				for train, test in skf.split(X, y):
					X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
					# labels = np.random.randint(0, n_clusters, [len(X_train)])
					labels = clustering.fit(X_train).labels_

					# Initialize new algorithm with k-means clusterings
					# best_clustering, clusters, models, lbls, errors, seps = optimized_sc(X_train, labels,\
					# n_iter, np.ravel(y_train), 5, 0.0, 0, dist="new", classifier="LR", verbose=True)
					# scores = score(X_test, np.ravel(y_test), models, clusters[1], 10, 1,  flag="normal", verbose=True)

					cluster_centers, models, alt_labels, errors, seps, loss = cac(X_train, labels, 10, np.ravel(y_train), alpha, -10, classifier="LR", verbose=True)
					scores = score(X_train, X_test, np.array(y_test), models, cluster_centers[1], alt_labels, alpha, flag="normal", verbose=True)[1:3]
					train_stats += np.vstack(cluster_centers[0]) 
					test_stats  += np.vstack([scores[0], scores[1]]).T

				train_stats /= 5
				test_stats /= 5
				results[key] = (train_stats, test_stats)
				with open('out4.py', 'a') as f:
					print(key + ": " + str(results[key]), file=f)
			flag = 0