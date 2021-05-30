from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, RidgeClassifier
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

scale = StandardScaler()
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################

color = ['grey', 'red', 'blue', 'pink', 'brown', 'black', 'magenta', 'purple', 'orange', 'cyan', 'olive']


class CAC(object):
    def __init__(self, n_clusters, alpha, beta=-np.infty, n_epochs=100, classifier="LR", decay="fixed", init="KM", verbose=False):
        self.k = n_clusters
        self.alpha = alpha
        self.beta = beta
        self.classifier = classifier
        self.decay = decay
        self.init = "KM"
        self.verbose = verbose
        self.n_epochs = n_epochs
        self.centers = []
        self.cluster_stats = []
        self.models = []
        self.scores = []

        self.clustering_loss = None
        self.classification_loss = []

    def fit(self, X, y):
        X = np.array(X)
        y = np.ravel(y)

        if self.init == "KM":
            clustering = KMeans(n_clusters=self.k, random_state=0, max_iter=300)
            cluster_labels = clustering.fit(X).labels_
        elif init == "RAND":
            cluster_labels = np.random.randint(0, self.k, [len(X)])

        label = []
        cluster_label = []
        y = np.array(y)
        N, d = X.shape
        labels = np.copy(cluster_labels)

        models = []
        lbls = []
        errors = np.zeros((self.n_epochs, self.k, 2))
        centers = np.zeros((self.k,d))
        positive_centers = np.zeros((self.k,d))
        negative_centers = np.zeros((self.k,d))
        cluster_stats = np.zeros((self.k,2))
        cac_classification_loss = []

        # initializing the mu arrays
        for j in range(self.k):
            pts_index = np.where(labels == j)[0]
            cluster_pts = X[pts_index]        
            centers[j,:] = cluster_pts.mean(axis=0)
            n_class_index = np.where(y[pts_index] == 0)[0]
            p_class_index = np.where(y[pts_index] == 1)[0]

            cluster_stats[j][0] = len(p_class_index)
            cluster_stats[j][1] = len(n_class_index)

            n_class = cluster_pts[n_class_index]
            p_class = cluster_pts[p_class_index]

            negative_centers[j,:] = n_class.mean(axis=0)
            positive_centers[j,:] = p_class.mean(axis=0)

        # initial performance
        scores, km_loss, km_models = self.evaluate_cac(X, y, labels)
        self.models.append(km_models)
        self.classification_loss.append(km_loss)
        self.scores.append(scores)
        self.centers.append(np.array([centers, positive_centers, negative_centers]))

        lbls.append(np.copy(labels))

        iteration = 0
        for idp in range(N):
            pt = X[idp]
            cluster_id = labels[idp]
            errors[iteration][cluster_id][0] += self.compute_euclidean_distance(pt, centers[cluster_id])
            errors[iteration][cluster_id][1] -= self.alpha*self.compute_euclidean_distance(positive_centers[cluster_id], negative_centers[cluster_id])

        cac_classification_loss.append(km_loss)

        for iteration in range(1, self.n_epochs):
            # print("iteration #", iteration-1)
            if self.decay == "inv":
                alpha_t = self.alpha*(iteration*0.1)/(1+iteration*0.1)
            elif self.decay == "fixed":
                alpha_t = self.alpha
            elif self.decay == "exp":
                alpha_t = self.alpha*(1-np.exp(-iteration/10))

            cluster_label = []
            for index_point in range(N):
                distance = {}
                pt = X[index_point]
                pt_label = y[index_point]
                cluster_id = labels[index_point]
                p, n = cluster_stats[cluster_id][0], cluster_stats[cluster_id][1]
                new_cluster = old_cluster = labels[index_point]

                # Ensure that degeneracy is not happening
                if ~((p == 1 and pt_label == 1) or (n == 1 and pt_label == 0)):
                    for cluster_id in range(0, self.k):
                        if cluster_id != old_cluster:
                            distance[cluster_id] = self.calculate_gamma_new(pt, pt_label,\
                                                    centers[cluster_id], positive_centers[cluster_id],\
                                                    negative_centers[cluster_id], cluster_stats[cluster_id], alpha_t)
                        else:
                            distance[cluster_id] = np.infty

                    old_gamma = self.calculate_gamma_old(pt, pt_label,\
                                                    centers[old_cluster], positive_centers[old_cluster],\
                                                    negative_centers[old_cluster], cluster_stats[old_cluster], alpha_t)

                    # new update condition
                    new_cluster = min(distance, key=distance.get)
                    new_gamma = distance[new_cluster]

                    if self.beta < old_gamma + new_gamma < 0:
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
                pt = X[idp]
                cluster_id = labels[idp]
                errors[iteration][cluster_id][0] += self.compute_euclidean_distance(pt, centers[cluster_id])
                errors[iteration][cluster_id][1] -= alpha_t*self.compute_euclidean_distance(positive_centers[cluster_id], negative_centers[cluster_id])

            # Store best clustering
            # print(np.sum(errors[iteration]))
            # print("***")
            # print(errors[iteration-1], np.sum(errors[iteration-1]))
            # if (np.abs(np.sum(errors[iteration]) - np.sum(errors[iteration-1])) < 0.01):
            # if verbose or (best[0][0] < f1 and best[0][1] < roc and (len(np.unique(labels)) == self.k)):
            lbls.append(np.copy(labels))
            if self.verbose == True:
                scores, loss, model = self.evaluate_cac(X, y, labels)
                self.scores.append(scores)
                self.centers.append(np.array([centers, positive_centers, negative_centers]))
                self.models.append(model)
                self.classification_loss.append(loss)

                if ((lbls[iteration] == lbls[iteration-1]).all()) and iteration > 0:
                    print("converged at itr: ", iteration)
                    break

            if ((lbls[iteration] == lbls[iteration-1]).all()) and iteration > 0:
                print("converged at itr: ", iteration)
                scores, loss, model = self.evaluate_cac(X, y, labels)
                self.scores.append(scores)
                self.centers.append(np.array([centers, positive_centers, negative_centers]))
                self.models.append(model)
                self.classification_loss.append(loss)
                break

        self.cluster_stats = cluster_stats
        self.clustering_loss = errors[:iteration+1]
        return None

    def get_base_model(self, classifier):
        if classifier == "LR":
            model = LogisticRegression(class_weight='balanced', random_state=0, max_iter=1000)
        elif classifier == "RF":
            model = RandomForestClassifier(n_estimators=10, random_state=0)
        elif classifier == "SVM":
            model = LinearSVC(max_iter = 10000)
            model.predict_proba = lambda X: np.array([model.decision_function(X), model.decision_function(X)]).transpose()
        elif classifier == "Perceptron":
            model = Perceptron()
            model.predict_proba = lambda X: np.array([model.decision_function(X), model.decision_function(X)]).transpose()
        elif classifier == "ADB":
            model = AdaBoostClassifier(n_estimators = 50)
        elif classifier == "DT":
            model = DecisionTreeClassifier()
        elif classifier == "LDA":
            model = LDA()
        elif classifier == "SGD":
            model = SGDClassifier(loss='log')
            # model.predict_proba = lambda X: np.array([model.decision_function(X), model.decision_function(X)]).transpose()
        elif classifier == "Ridge":
            model = RidgeClassifier()
            model.predict_proba = lambda X: np.array([model.decision_function(X), model.decision_function(X)]).transpose()
        elif classifier == "NB":
            model = MultinomialNB()
        elif classifier == "KNN":
            model = KNeighborsClassifier(n_neighbors=5)
        else:
            model = LogisticRegression(class_weight='balanced', random_state=0, max_iter=1000)
        return model

    def evaluate_cac(self, X, y, cluster_labels):
        y_pred = []
        y_true = []
        y_proba = []
        models = []
        loss = []

        for j in np.unique(cluster_labels):
            model = self.get_base_model(self.classifier)
            pts_index = np.where(cluster_labels == j)[0]
            x_cluster = X[pts_index]
            y_cluster = y[pts_index]

            clf = model.fit(x_cluster, y_cluster)
            pred = clf.predict(x_cluster)
            y_predict_proba = clf.predict_proba(x_cluster)

            y_pred.extend(pred)
            y_true.extend(y_cluster)
            y_proba.extend(y_predict_proba[:,1])
            # models.append([model, self.best_threshold(y_cluster, y_predict_proba[:,1])])
            models.append(model)
            loss.append(log_loss(y_cluster, y_predict_proba))

        acc = metrics.accuracy_score(y_true, y_pred)
        f1 = f1_score(y_pred, y_true)
        auc = roc_auc_score(y_true, y_proba)
        params = confusion_matrix(y_true, y_pred).reshape(4,1)
        spe = self.specificity(params)[0]
        sen = self.sensitivity(params)[0]
        scores = [acc, f1, auc, spe, sen]
        return scores, sum(loss), models


    def predict(self, X_test, iteration):
        order = []
        # for j in range(len(self.models)):
        model = self.models[iteration]
        predicted_clusters = self.predict_clusters(X_test, self.centers[iteration])
        pred = []
        pred_proba = []
        new_y_test = []
        K = np.unique(predicted_clusters)
        N = len(X_test)

        for c in K:
            m = model[int(c)]
            # thresh = model[int(c)][1]
            cluster_point_index = np.where(predicted_clusters == c)[0]
            order += list(cluster_point_index)
            # new_y_test.extend(y_test[cluster_point_index])
            cluster_pred_proba = m.predict_proba(X_test[cluster_point_index])[:,1]
            # cluster_preds = 1*(cluster_pred_proba >= thresh)
            # # print(thresh, best_threshold(y_test[cluster_point_index], cluster_pred_proba))
            pred.extend(m.predict(X_test[cluster_point_index]))
            # pred.extend(cluster_preds)
            pred_proba.extend(cluster_pred_proba)

        y_pred = np.zeros(N)
        y_pred_proba = np.zeros(N)
        for i, j in enumerate(order):
            y_pred[j] = pred[i]
            y_pred_proba[j] = pred_proba[i]
        return y_pred, y_pred_proba
        # return np.array([np.array(acc[1:]), np.array(f1[1:]), np.array(roc[1:]), np.array(spe[1:]), np.array(sen[1:])], dtype=object), log_loss(y_test, pred_proba)


    def best_threshold(self, y_test, y_predict_proba):
        fpr, tpr, thresholds = roc_curve(y_test, y_predict_proba)
        # # get the best threshold
        J = tpr - fpr
        ix = np.argmax(J)
        best_thresh = thresholds[ix]
        return best_thresh


    def predict_clusters(self, X_test, all_centers) -> np.array:
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


    def compute_euclidean_distance(self, point, centroid):
        # return np.sum((point - centroid)**2)
        return np.sum((point - centroid)**2)


    def calculate_gamma_old(self, pt, label, mu, mup, mun, cluster_stats, alpha=2):
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

        arr1 = np.array([mup_new, mup, mu_new])
        arr2 = np.array([mun_new, mun, mu])
        diff = arr1 - arr2

        vals = np.sum(np.square(diff), axis=1)

        # new_lin_sep = np.sum(np.square(mun_new - mup_new))
        # lin_sep = np.sum(np.square(mun - mup))
        # mu_sep = np.sum(np.square(mu - mu_new))

        new_lin_sep = vals[0]
        lin_sep = vals[1]
        mu_sep = vals[2]

        gamma_p = -np.sum(np.square(mu-pt)) - (p+n-1) * mu_sep + (p+n) * alpha*lin_sep - (p+n-1)*alpha*new_lin_sep
        # gamma_p = -np.sum(np.square(mu-pt)) - (p+n-1) * mu_sep + alpha*lin_sep - alpha*new_lin_sep
        return gamma_p


    def calculate_gamma_new(self, pt, label, mu, mup, mun, cluster_stats, alpha=2):
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

        # new_lin_sep = np.sum(np.square(mun_new - mup_new))
        # lin_sep = np.sum(np.square(mun - mup))
        # mu_sep = np.sum(np.square(mu - mu_new))
        arr1 = np.array([mup_new, mup, mu_new])
        arr2 = np.array([mun_new, mun, mu])
        diff = arr1 - arr2
        vals = np.sum(np.square(diff), axis=1)

        new_lin_sep = vals[0]
        lin_sep = vals[1]
        mu_sep = vals[2]

        gamma_j = np.sum(np.square(mu_new-pt)) + (p+n)*mu_sep + (p+n) * alpha*lin_sep - (p+n+1)*alpha*new_lin_sep
        # gamma_j = np.sum(np.square(mu_new-pt)) + (p+n)*mu_sep + alpha*lin_sep - alpha*new_lin_sep
        return gamma_j

    def specificity(self, params):
        return (params[0]/(params[0]+params[1]))

    def sensitivity(self, params):
        return (params[3]/(params[3]+params[2]))
