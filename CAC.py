from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn import model_selection, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.cluster import KMeans
from typing import Tuple
import pandas as pd
import numpy as np
import sys
import umap

# tn, fp, fn, tp
def specificity(params):
    return (params[0]/(params[0]+params[1]))

def sensitivity(params):
    return (params[3]/(params[3]+params[2]))

scale = StandardScaler()
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################

color = ['grey', 'red', 'blue', 'pink', 'brown', 'black', 'magenta', 'purple', 'orange', 'cyan', 'olive']

def best_threshold(y_test, y_predict_proba):
    fpr, tpr, thresholds = roc_curve(y_test, y_predict_proba)
    # # get the best threshold
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    return best_thresh


def predict_clusters(X_test, all_centers, alpha) -> np.array:
    if len(all_centers) == 1:
        centers = all_centers
    elif len(all_centers) == 3:
        centers, p_centers, n_centers = all_centers

    K = len(centers)
    dists = np.zeros(K)
    test_labels = np.zeros(X_test.shape[0])
    label = 1

    for pt in range(X_test.shape[0]):
        for k in range(K):
            min_dist = np.square(np.linalg.norm(centers[k] - X_test[pt]))
            dists[k] = min_dist
        test_labels[pt] = np.argmin(dists)

    return test_labels


def predict_clusters_cac(X_test, all_centers, labels, alpha) -> np.array:
    if len(all_centers) == 1:
        centers = all_centers
    elif len(all_centers) == 3:
        centers, p_centers, n_centers = all_centers

    K = len(centers)
    dists = np.zeros(K)
    test_labels = np.zeros(X_test.shape[0])
    C = []
    for i in range(K):
        C.append(len(np.where(labels == i)[0]))

    for pt in range(X_test.shape[0]):
        for k in range(K):
            min_dist = np.square(np.linalg.norm(centers[k] - X_test[pt]))
            min_dist -= alpha*np.square(np.linalg.norm(p_centers[k] - n_centers[k]))
            dists[k] = min_dist
        test_labels[pt] = np.argmin(dists)
    return test_labels


def compute_euclidean_distance(point, centroid):
    # return np.sum((point - centroid)**2)
    return np.sum((point - centroid)**2)


def calculate_gamma_old(pt, label, mu, mup, mun, cluster_stats, alpha=2):
    p, n = cluster_stats[0], cluster_stats[1]
    if label == 0:
        mun_new = (n/(n-1))*mun - (1/(n-1))*pt
        mup_new = mup
        n_new = n-1
        p_new = p
    else:
        mup_new = (p/(p-1))*mup - (1/(p-1))*pt
        mun_new = mun
        p_new = p-1
        n_new = n

    mu_new = (p_new*mup_new + n_new*mun_new)/(p_new + n_new)

    new_lin_sep = np.sum(np.square(mun_new - mup_new))
    lin_sep = np.sum(np.square(mun - mup))
    mu_sep = np.sum(np.square(mu - mu_new))
    gamma_p = -np.sum(np.square(mu-pt)) - (p+n-1) * mu_sep + (p+n) * alpha*lin_sep - (p+n-1)*alpha*new_lin_sep
    # gamma_p = -np.sum(np.square(mu-pt)) - (p+n-1) * mu_sep + alpha*lin_sep - alpha*new_lin_sep
    return gamma_p


def calculate_gamma_new(pt, label, mu, mup, mun, cluster_stats, alpha=2):
    p, n = cluster_stats[0], cluster_stats[1]
    if label == 0:
        mun_new = (n/(n+1))*mun + (1/(n+1))*pt
        mup_new = mup
        n_new = n+1
        p_new = p

    else:
        mup_new = (p/(p+1))*mup + (1/(p+1))*pt
        mun_new = mun
        p_new = p+1
        n_new = n

    mu_new = (p_new*mup_new + n_new*mun_new)/(p_new + n_new)
    new_lin_sep = np.sum(np.square(mun_new - mup_new))
    lin_sep = np.sum(np.square(mun - mup))
    mu_sep = np.sum(np.square(mu - mu_new))

    gamma_j = np.sum(np.square(mu_new-pt)) + (p+n)*mu_sep + (p+n) * alpha*lin_sep - (p+n+1)*alpha*new_lin_sep
    # gamma_j = np.sum(np.square(mu_new-pt)) + (p+n)*mu_sep + alpha*lin_sep - alpha*new_lin_sep
    return gamma_j


def cac(data_points, cluster_labels, total_iteration, y, alpha, beta, classifier="LR", verbose=False):
    label = []
    cluster_label = []
    y = np.array(y)
    N, d = data_points.shape
    k = len(np.unique(cluster_labels))
    labels = np.copy(cluster_labels)

    best = [[], []]
    models = []
    lbls = []
    errors = np.zeros((total_iteration, k, 2))
    centers = np.zeros((k,d))
    positive_centers = np.zeros((k,d))
    negative_centers = np.zeros((k,d))
    cluster_stats = np.zeros((k,2))
    seps = []
    loss = []

    # Initializing the mu arrays
    for j in range(k):
        pts_index = np.where(labels == j)[0]
        cluster_pts = data_points[pts_index]        
        centers[j,:] = cluster_pts.mean(axis=0)
        n_class_index = np.where(y[pts_index] == 0)[0]
        p_class_index = np.where(y[pts_index] == 1)[0]

        cluster_stats[j][0] = len(p_class_index)
        cluster_stats[j][1] = len(n_class_index)

        n_class = cluster_pts[n_class_index]
        p_class = cluster_pts[p_class_index]

        negative_centers[j,:] = n_class.mean(axis=0)
        positive_centers[j,:] = p_class.mean(axis=0)

    # Initial performance
    f1, roc, m, l = get_new_accuracy(data_points, labels, y, classifier)
    best[0].append(np.array([f1, roc]))
    best[1].append(np.array([centers, positive_centers, negative_centers]))
    models.append(m)
    lbls.append(np.copy(labels))
    s = 0
    for clstr in range(k):
        sep = compute_euclidean_distance(negative_centers[clstr], positive_centers[clstr])
        s += sep
    seps.append(s)
    loss.append(l)

    for iteration in range(0, total_iteration):
        # print("iteration #", iteration-1)
        # alpha_t = alpha/(1+iteration*0.1)
        alpha_t = alpha*(iteration*0.1)/(1+iteration*0.1)
        # alpha_t = alpha*(1-np.exp(-iteration/10))
        # alpha_t = alpha*(1-np.power(0.5, np.floor(iteration/5)))
        # alpha_t = alpha
        cluster_label = []
        for index_point in range(N):
            distance = {}
            pt = data_points[index_point]
            pt_label = y[index_point]
            cluster_id = labels[index_point]
            p, n = cluster_stats[cluster_id][0], cluster_stats[cluster_id][1]
            new_cluster = old_cluster = labels[index_point]
            old_err = np.zeros(k)
            # Ensure that degeneracy is not happening
            if ~((p == 1 and pt_label == 1) or (n == 1 and pt_label == 0)):
                # # print("Considering changing label of point ", index_point)
                for cluster_id in range(0, k):
                    if cluster_id != old_cluster:
                        distance[cluster_id] = calculate_gamma_new(pt, pt_label,\
                                                centers[cluster_id], positive_centers[cluster_id],\
                                                negative_centers[cluster_id], cluster_stats[cluster_id], alpha_t)
                    else:
                        distance[cluster_id] = np.infty

                old_gamma = calculate_gamma_old(pt, pt_label,\
                                                centers[old_cluster], positive_centers[old_cluster],\
                                                negative_centers[old_cluster], cluster_stats[old_cluster], alpha_t)
                # new update condition
                new_cluster = min(distance, key=distance.get)
                new_gamma = distance[new_cluster]

                if beta < old_gamma + new_gamma < 0:
                    # Remove point from old cluster
                    p, n = cluster_stats[old_cluster] # Old cluster statistics
                    t = p + n

                    centers[old_cluster] = (t/(t-1))*centers[old_cluster] - (1/(t-1))*pt

                    if pt_label == 0:
                        negative_centers[old_cluster] = (n/(n-1))*negative_centers[old_cluster] - (1/(n-1)) * pt
                        cluster_stats[old_cluster][1] -= 1

                    else:
                        positive_centers[old_cluster] = (p/(p-1))*positive_centers[old_cluster] - (1/(p-1)) * pt
                        cluster_stats[old_cluster][0] -= 1

                    # Add point to new cluster
                    p, n = cluster_stats[new_cluster] # New cluster statistics
                    t = p + n
                    centers[new_cluster] = (t/(t+1))*centers[new_cluster] + (1/(t+1))*pt

                    if pt_label == 0:
                        negative_centers[new_cluster] = (n/(n+1))*negative_centers[new_cluster] + (1/(n+1)) * pt
                        cluster_stats[new_cluster][1] += 1

                    else:
                        positive_centers[new_cluster] = (p/(p+1))*positive_centers[new_cluster] + (1/(p+1)) * pt
                        cluster_stats[new_cluster][0] += 1
                    labels[index_point] = new_cluster

        for idp in range(N):
            pt = data_points[idp]
            cluster_id = labels[idp]
            # errors[iteration][cluster_id] += compute_euclidean_distance(pt, centers[cluster_id])-alpha_t*compute_euclidean_distance(positive_centers[cluster_id],\
                                                # negative_centers[cluster_id])
            errors[iteration][cluster_id][0] += compute_euclidean_distance(pt, centers[cluster_id])
            errors[iteration][cluster_id][1] -= alpha_t*compute_euclidean_distance(positive_centers[cluster_id], negative_centers[cluster_id])

        # Store best clustering
        f1, roc, m, l = get_new_accuracy(data_points, labels, y, classifier)
        # if verbose or (best[0][0] < f1 and best[0][1] < roc and (len(np.unique(labels)) == k)):
        best[0].append(np.array([f1, roc]))
        best[1].append(np.array([centers, positive_centers, negative_centers]))
        models.append(m)
        lbls.append(np.copy(labels))
        seps.append(s)
        loss.append(l)
        # print(np.sum(errors[iteration]))
        # print("***")
        # print(errors[iteration-1], np.sum(errors[iteration-1]))
        # if (np.abs(np.sum(errors[iteration]) - np.sum(errors[iteration-1])) < 0.01):
        if ((lbls[iteration] == lbls[iteration-1]).all()) and iteration > 0:
            print("converged at itr: ", iteration)
            break
    # print("Errors at iteration #", iteration)
    # print(errors)
    # print(np.sum(errors, axis=1)[:,0])
    return np.array(best), models, lbls, errors, seps, loss


def get_new_accuracy(X, cluster_labels, y, classifier):
    y_pred = []
    y_true = []
    y_proba = []
    models = []
    loss = []

    for j in np.unique(cluster_labels):
        if classifier == "LR":
            model = LogisticRegression(class_weight='balanced', random_state=0, max_iter=1000)
        elif classifier == "RF":
            model = RandomForestClassifier(n_estimators=10, random_state=0)
        elif classifier == "SVM":
            model = SVC(kernel="linear", probability=True)
            # model = LinearSVC(max_iter = 1000)
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

        pts_index = np.where(cluster_labels == j)[0]
        x_cluster = X[pts_index]
        y_cluster = y[pts_index]

        clf = model.fit(x_cluster, y_cluster)
        pred = clf.predict(x_cluster)
        y_predict_proba = clf.predict_proba(x_cluster)

        y_pred.extend(pred)
        y_true.extend(y_cluster)
        y_proba.extend(y_predict_proba[:,1])
        models.append([model, best_threshold(y_cluster, y_predict_proba[:,1])])
        loss.append(log_loss(y_cluster, y_predict_proba))

    # print("\nTraining Data Cluster Classification Metrics:\n")
    # print("F1 " + str(metrics.f1_score(y_pred, y_true)))
    # print("ROC " + str(metrics.roc_auc_score(y_true, y_proba)))

    # # print("=============\n")
    return metrics.f1_score(y_pred, y_true), metrics.roc_auc_score(y_true, y_proba), models, sum(loss)


def score(X_test, y_test, models, clusters, labels, alpha, flag="none", verbose=False):
    acc, f1, roc, spe, sen = [0],[0],[0],[0],[0]
    for j in range(len(models)):
        model = models[j]
        if flag == "cac":
            predicted_clusters = predict_clusters_cac(X_test, clusters[j], labels[j], alpha)
        else:
            predicted_clusters = predict_clusters(X_test, clusters[j], alpha)
        pred = []
        pred_proba = []
        new_y_test = []
        K = np.unique(predicted_clusters)
        for c in K:
            m = model[int(c)][0]
            thresh = model[int(c)][1]
            cluster_point_index = np.where(predicted_clusters == c)[0]

            new_y_test.extend(y_test[cluster_point_index])
            cluster_pred_proba = m.predict_proba(X_test[cluster_point_index])[:,1]
            # cluster_preds = 1*(cluster_pred_proba >= thresh)
            # # print(thresh, best_threshold(y_test[cluster_point_index], cluster_pred_proba))
            pred.extend(m.predict(X_test[cluster_point_index]))
            # pred.extend(cluster_preds)
            pred_proba.extend(cluster_pred_proba)

        if (f1_score(pred, new_y_test) > max(f1) or roc_auc_score(new_y_test, pred_proba) > max(roc)) or verbose:
            acc.append(metrics.accuracy_score(new_y_test, pred))
            f1.append(f1_score(pred, new_y_test))
            roc.append(roc_auc_score(new_y_test, pred_proba))
            params = confusion_matrix(new_y_test, pred).reshape(4,1)
            spe.append(specificity(params)[0])
            sen.append(sensitivity(params)[0])
            # if verbose:
                # if j == 0:
                    # print("Test acuracy for normal k-means clusters")
                # else:
                    # print("Test Accuracy for clustering " + str(j))
                # print("Accuracy: " + str(acc[-1]))
                # print("F1 score: " + str(f1[-1]))
                # print("RoC score: " + str(roc[-1]))
                # print("Specificity: " + str(spe[-1]))
                # print("Sensitivity: " + str(sen[-1]))
                # print("======\n")
    return np.array([np.array(acc[1:]), np.array(f1[1:]), np.array(roc[1:]), np.array(spe[1:]), np.array(sen[1:])])
