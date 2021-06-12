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

test_results = pd.DataFrame(columns=['Dataset', 'Classifier', 'alpha', 'k', \
    'Base_F1_mean', 'Base_AUC_mean',\
    'KM_F1_mean', 'KM_AUC_mean', \
    'CAC_F1_mean', 'CAC_AUC_mean'], dtype=object)

res = pd.DataFrame(columns=['Dataset', 'Classifier', 'alpha', 'k', \
    'Base_F1_mean', 'Base_AUC_mean',\
    'KM_F1_mean', 'KM_AUC_mean', \
    'CAC_F1_mean', 'CAC_AUC_mean'], dtype=object)

# DATASET = "sepsis" # see folder, *the Titanic dataset is different*
DATASET = args.dataset

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

params = {'titanic': 0.2,
  'magic': 0.01,
  'creditcard': 0.3,
  'adult': 0.8,
  'diabetes': 0.15,
  'cic': 0.5}

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

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=108)

def neural_network(n_experts, cluster_algo, alpha):
  beta = -np.infty # do not change this
  precision_lst = []
  recall_lst = []
  fscore_lst = []
  auroc_lst = []
  accuracy_lst = []

  for train, test in skf.split(X, y):
      X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
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
      # plot_model(dae, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
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
      precision_lst.append(scores[0])
      recall_lst.append(scores[1])
      fscore_lst.append(f1_score(y_test, y_pred > 0.5))
      accuracy_lst.append(accuracy_score(y_test, y_pred > 0.5))

      if len(np.unique(y_test)) == 1:
        auroc_lst.append(0)
      else:
        auroc_lst.append(roc_auc_score(y_test, y_pred))

      # print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred > 0.5))
      backend.clear_session()

      # print('Precision: ', precision_lst[-1])
      # print('Recall: ', recall_lst[-1])
      # print('F1 score: ', fscore_lst[-1])
      # print('AUROC: ', auroc_lst[-1])
      # print('Accuracy: ', accuracy_lst[-1])

  return {'precision': np.mean(precision_lst),
          'recall': np.mean(recall_lst),
          'f1_score': np.mean(fscore_lst),
          'auroc': np.mean(auroc_lst),
          'accuracy': np.mean(accuracy_lst)}

test_results = pd.DataFrame(columns=['Dataset', 'Classifier', 'alpha', 'k', \
    'Base_F1_mean', 'Base_AUC_mean',\
    'KM_F1_mean', 'KM_AUC_mean', \
    'CAC_F1_mean', 'CAC_AUC_mean'], dtype=object)

alpha = params[DATASET]
k = n_cluster_params[DATASET]
scores_base = neural_network(1, 'KMeans', alpha)
scores_km = neural_network(k, 'KMeans', alpha)
scores_cac = neural_network(k, 'CAC', alpha)

test_f1_auc = [scores_base['f1_score'], scores_base['auroc'], scores_km['f1_score'], scores_km['auroc'], scores_cac['f1_score'], scores_cac['auroc']]
test_results.loc[0] = [DATASET, "DMNN", alpha, k] + test_f1_auc

test_results.to_csv("./Results/5CV_DMNN_Results_" + args.dataset + ".csv", index=None)