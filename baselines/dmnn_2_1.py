import pandas as pd
import numpy as np
import argparse
import os
import sys
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import itertools
import random
from sklearn.metrics import classification_report, roc_curve, confusion_matrix, precision_recall_fscore_support, roc_auc_score, accuracy_score, f1_score
from sklearn.utils import class_weight, shuffle
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from keras import models, layers, losses, optimizers, initializers, regularizers
from keras.utils.vis_utils import plot_model
from keras import backend
import matplotlib.pyplot as plt

from CAC import specificity, sensitivity, best_threshold, predict_clusters, predict_clusters_cac,\
compute_euclidean_distance, calculate_gamma_old, calculate_gamma_new,\
cac, get_new_accuracy, score


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ALL')
parser.add_argument('--init', default='RAND')
parser.add_argument('--classifier', default='ALL')
parser.add_argument('--verbose', default="False")
parser.add_argument('--decay', default='Fixed')
parser.add_argument('--alpha')
parser.add_argument('--k')
parser.add_argument('--cv', default="False")
args = parser.parse_args()

test_results = pd.DataFrame(columns=['Dataset', 'Classifier', 'alpha', 'k', \
    'Base_F1_mean', 'Base_AUC_mean','Base_F1_std', 'Base_AUC_std',\
    'KM_F1_mean', 'KM_AUC_mean', 'KM_F1_std', 'KM_AUC_std',\
    'CAC_F1_mean', 'CAC_AUC_mean', 'CAC_F1_std', 'CAC_AUC_std'], dtype=object)

res = pd.DataFrame(columns=['Dataset', 'Classifier', 'alpha', 'k', \
    'Base_F1_mean', 'Base_AUC_mean',\
    'KM_F1_mean', 'KM_AUC_mean', \
    'CAC_F1_mean', 'CAC_AUC_mean'], dtype=object)

# DATASET = "sepsis" # see folder, *the Titanic dataset is different*
DATASET = args.dataset
alphas = [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1]
############ FOR CIC DATASET ############
if DATASET == "cic":
    Xa = pd.read_csv("./data/cic/cic_set_a.csv")
    Xb = pd.read_csv("./data/cic/cic_set_b.csv")
    Xc = pd.read_csv("./data/cic/cic_set_c.csv")

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


def neural_network(X_train, y_train, X_test, y_test, n_experts, cluster_algo, alpha):
    # alpha = 0.04 # change this value only
    beta = -np.infty # do not change this
    data_len = X_train.shape[1]
    ## Define Neural Network
    experts = []
    inputTensor = layers.Input(shape=(data_len,))
    encode = layers.Dense(units=64, name='encode_1', activation=None, activity_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(inputTensor)
    encode = layers.Dense(units=32, name='encode_2', activation=None, activity_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(encode)
    decode = layers.Dense(units=64, name='decode_2', activation=None)(encode)
    decode = layers.Dense(units=data_len, name='decode_1', activation=None)(decode)

    gate = layers.Dense(n_experts, activation='softmax', name='gating')(encode)

    for i in range(n_experts):
      layer_var = layers.Dense(30, activation='relu', name='dense_{}_2'.format(i))(encode)
      layer_var = layers.Dense(1, activation='sigmoid', name='dense_{}_3'.format(i))(layer_var)
      experts.append(layer_var)
      del layer_var

    if n_experts == 1:
      outputTensor = experts
    else: 
      mergedTensor = layers.Concatenate(axis=1)(experts)
      outputTensor = layers.Dot(axes=1)([gate, mergedTensor])

    # Define autoencoder
    dae = models.Model(inputs=inputTensor, outputs=decode)
    dae.compile(
        optimizer=optimizers.Adadelta(learning_rate=0.1),
        loss='MeanSquaredError'
    )

    # Define cluster gating
    cluster = models.Model(inputs=inputTensor, outputs=gate)
    for i in ['encode_1', 'encode_2']:
      cluster.get_layer(i).trainable = False
    # print(cluster.summary())
    # plot_model(cluster, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    cluster.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
    )

    # Define full model
    full = models.Model(inputs=inputTensor, outputs=outputTensor)
    for i in ['encode_1', 'encode_2']:
      full.get_layer(i).trainable = True
    # print(full.summary())
    # plot_model(full, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    full.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['binary_crossentropy'],
    )

    ## Train autoencoder
    history_dae = dae.fit(
        x=X_train+np.random.normal(0,0.05,X_train.shape),
        y=X_train,
        batch_size=1024,
        shuffle=True,
        epochs=100,
        use_multiprocessing=True,
        verbose=0
        )
    # print("AutoEncoder Trained")

    ## Check autoencoder loss
    # plt.plot(history_dae.history['loss'], label='Training loss')
    # plt.title('Loss')
    # plt.ylabel('Loss')
    # plt.xlabel('No. epoch')
    # plt.legend(loc="best")
    # plt.show()
    test_loss = dae.evaluate(x=X_test, y=X_test)
    # print('Autoencoder loss:', test_loss)

    ## Get embeddings
    encoder = models.Model(inputs=inputTensor, outputs=encode)
    X_train_embeddings = encoder.predict(x=X_train)

    # X_train_embeddings_2 = TSNE(n_components=2, verbose=1, n_jobs=-1).fit_transform(X_train_embeddings[:2000])
    
    if cluster_algo == 'KMeans':
      ## KMeans Clustering
      cluster_alg = KMeans(n_clusters=n_experts, random_state=0)
      X_train_clusters = cluster_alg.fit_predict(X_train_embeddings)
    elif cluster_algo == 'CAC':
      ## CAC Clustering
      labels = np.random.randint(0, n_experts, [len(X_train)])
      _, _, alt_labels, _, _, _ = cac(X_train, labels, 100, np.ravel(y_train), alpha, beta, classifier="LR", verbose=True)
      X_train_clusters = alt_labels[-1]
    else:
      raise ValueError('Method not supported')

    # plt.figure(figsize=(10,10))
    # for i in range(n_experts):
    #   plt.plot(X_train_embeddings_2[X_train_clusters[:2000]==i,0], X_train_embeddings_2[X_train_clusters[:2000]==i,1], '.')
    # plt.title('TSNE')
    # plt.show()

    X_train_clusters_sparse = MultiLabelBinarizer().fit_transform(X_train_clusters.reshape(-1,1))

    ## Train cluster gating
    history_cluster = cluster.fit(
        x=X_train,
        y=X_train_clusters_sparse,
        batch_size=1000,
        shuffle=True,
        epochs=100,
        use_multiprocessing=True,
        verbose=0
        )

    ## Check cluster gating accuracy
    y_pred = cluster.predict(X_train) 
    scores = precision_recall_fscore_support(X_train_clusters_sparse, y_pred > 0.5, average='weighted')
    # print('Precision (cluster): ', scores[0])
    # print('Recall (cluster): ', scores[1])
    # print('F1 score (cluster): ', f1_score(X_train_clusters_sparse, y_pred > 0.5, average='micro'))
    # print('Accuracy (cluster): ', accuracy_score(X_train_clusters_sparse, y_pred > 0.5))
    # if n_experts < 2:
    #   print('AUROC (cluster): ', 'NIL')
    # else:
    #   print('AUROC (cluster): ', roc_auc_score(X_train_clusters_sparse, y_pred))


    ## Train full model
    history_full = full.fit(
        x=X_train,
        y=y_train,
        batch_size=1000,
        shuffle=True,
        epochs=100,
        use_multiprocessing=True,
        verbose=0
        )

    ## Check full model accuracy
    y_pred = full.predict(X_test)

    scores = precision_recall_fscore_support(y_test, y_pred > 0.5, average='weighted')
    precision_ls = scores[0]
    recall_ls = scores[1]
    fscore_ls = f1_score(y_test, y_pred > 0.5)
    accuracy_ls = accuracy_score(y_test, y_pred > 0.5)

    if len(np.unique(y_test)) == 1:
      auroc_ls = 0
    else:
      auroc_ls = roc_auc_score(y_test, y_pred)

    # print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred > 0.5))
    backend.clear_session()

    # print('Precision: ', precision_ls)
    # print('Recall: ', recall_ls)
    # print('F1 score: ', fscore_ls)
    # print('AUROC: ', auroc_ls)
    # print('Accuracy: ', accuracy_ls)

    return {'precision': precision_ls,
          'recall': recall_ls,
          'f1_score': fscore_ls,
          'auroc': auroc_ls,
          'accuracy': accuracy_ls}


param_grid = {
'alpha': alphas,
# 'k': [2, 3, 4, 5]
'k': [2]
}


params = {'titanic': [0.2, 2],
  'magic': [0.01, 2],
  'creditcard': [0.3, 3],
  'adult': [0.8, 2],
  'diabetes': [0.15, 2],
  'cic': [0.1, 2]
}


best_alpha = 0
best_score = 0
test_f1_auc = [0, 0, 0, 0, 0, 0]
keys, values = zip(*param_grid.items())
combs = list(itertools.product(*values))
# random.shuffle(combs)
n_splits = 5
scale = StandardScaler()
res_idx = 0

if args.cv == "False":
    print("Testing on HELD OUT test set (5 times) with best alpha")
    X1, X_test, y1, y_test = train_test_split(X, y, stratify=y, random_state=108)
    alpha = params[args.dataset][0]
    n_clusters = params[args.dataset][1]

    X1 = scale.fit_transform(X1)
    X_test = scale.fit_transform(X_test)

    print("Testing on ", DATASET, " with alpha = ", alpha)

    n_runs = 5
    base_scores = np.zeros((n_runs, 2))
    km_scores = np.zeros((n_runs, 2))
    cac_scores = np.zeros((n_runs, 2))

    for i in range(n_runs):
        scores_cac = neural_network(X1, y1, X_test, y_test, n_clusters, 'CAC', alpha)
        scores_base = neural_network(X1, y1, X_test, y_test, 1, 'KMeans', alpha)
        scores_km = neural_network(X1, y1, X_test, y_test, n_clusters, 'KMeans', alpha)
                
        cac_scores[i, 0] = scores_cac['f1_score']
        cac_scores[i, 1] = scores_cac['auroc']
        km_scores[i, 0] = scores_km['f1_score']
        km_scores[i, 1] = scores_base['auroc']
        base_scores[i, 0] = scores_base['f1_score']
        base_scores[i, 1] = scores_base['auroc']

    print("Base final test performance: ", "F1: ", scores_base['f1_score'], "AUC: ", scores_base['auroc'], alpha)
    print("KM final test performance: ", "F1: ", scores_km['f1_score'], "AUC: ", scores_km['auroc'], alpha)
    print("CAC final test performance: ", "F1: ", scores_cac['f1_score'], "AUC: ", scores_cac['auroc'], alpha)
    test_results.loc[0] = [DATASET, "DMNN", alpha, n_clusters] + list(np.mean(base_scores, axis=0)) + list(np.std(base_scores, axis=0)) + \
            list(np.mean(km_scores, axis=0)) + list(np.std(km_scores, axis=0)) + \
            list(np.mean(cac_scores, axis=0)) + list(np.std(cac_scores, axis=0))
    print(test_results)
    test_results.to_csv("./Results/Best_alpha_test_" + args.dataset + "" + ".csv", index=None)

else:
    for v in combs:
        hyperparameters = dict(zip(keys, v)) 
        alpha = hyperparameters['alpha']
        n_clusters = params[DATASET][1] # Fix number of clusters to previous values
        print(hyperparameters)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=108)

        base_scores = np.zeros((n_splits, 2))
        km_scores = np.zeros((n_splits, 2))
        cac_scores = np.zeros((n_splits, 2))
        i = 0

        X1, X_test, y1, y_test = train_test_split(X, y, stratify=y, random_state=108)

        # Grid CV happening here
        print("Starting GridCV search")
        for train, val in skf.split(X1, y1):
            print("Iteration: ", i)
            X_train, X_val, y_train, y_val = X1[train], X1[val], y1[train], y1[val]
            X_train = scale.fit_transform(X_train)
            X_val = scale.fit_transform(X_val)

            scores_base = neural_network(X_train, y_train, X_test, y_test, 1, 'KMeans', alpha)
            scores_cac = neural_network(X_train, y_train, X_test, y_test, n_clusters, 'CAC', alpha)
            scores_km = neural_network(X_train, y_train, X_test, y_test, n_clusters, 'KMeans', alpha)

            base_scores[i, 0] = scores_base['f1_score']
            base_scores[i, 1] = scores_base['auroc']

            km_scores[i, 0] = scores_km['f1_score']
            km_scores[i, 1] = scores_km['auroc']

            cac_scores[i, 0] = scores_cac['f1_score']
            cac_scores[i, 1] = scores_cac['auroc']
            i += 1

        print("5-Fold Base scores", np.mean(base_scores, axis=0))
        print("5-Fold KMeans scores", np.mean(km_scores, axis=0))        
        print("5-Fold terminal CAC scores", np.mean(cac_scores, axis=0))
        print("\n")

        res.loc[res_idx] = [DATASET, "DMNN", alpha, n_clusters] + list(np.mean(base_scores, axis=0)) + \
        list(np.mean(km_scores, axis=0)) + \
        list(np.mean(cac_scores, axis=0))
        res_idx += 1
        res.to_csv("./Results/Tuning_every_run" + args.dataset + ".csv", index=None)

        X1 = scale.fit_transform(X1)
        X_test = scale.fit_transform(X_test)

        print("Testing on Test data with alpha = ", alpha)

        scores_cac = neural_network(X1, y1, X_test, y_test, n_clusters, 'CAC', alpha)
        scores_base = neural_network(X1, y1, X_test, y_test, 1, 'KMeans', alpha)
        scores_km = neural_network(X1, y1, X_test, y_test, n_clusters, 'KMeans', alpha)

        print("Base final test performance: ", "F1: ", scores_base['f1_score'], "AUC: ", scores_base['auroc'], alpha)
        print("KM final test performance: ", "F1: ", scores_km['f1_score'], "AUC: ", scores_km['auroc'], alpha)
        print("CAC final test performance: ", "F1: ", scores_cac['f1_score'], "AUC: ", scores_cac['auroc'], alpha)
        print("\n")

        # Can choose whether it to do it w.r.t F1 or AUC
        if np.mean(cac_scores, axis=0)[0] > best_score:
            best_score = np.mean(cac_scores, axis=0)[0]
            best_alpha = alpha
            best_k = n_clusters
            test_f1_auc = [scores_base['f1_score'], scores_base['auroc'], scores_km['f1_score'], scores_km['auroc'], scores_cac['f1_score'], scores_cac['auroc']]
            # print(test_f1_auc)
        
    print(DATASET, ": Best alpha = ", best_alpha)
    test_results.loc[0] = [DATASET, "DMNN", best_alpha, best_k] + test_f1_auc
    print(test_results)
    test_results.to_csv("./Results/Tuned_Test_Results_" + args.dataset + "" + ".csv", index=None)

    # result_1 = neural_network(1, 'KMeans')
    # result_2 = neural_network(2, 'KMeans')
    # result_3 = neural_network(2, 'CAC')

    # print(result_1)
    # print(result_2)
    # print(result_3)

