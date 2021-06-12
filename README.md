CAC
======

CAC is a clustering based framework for classification. The framework proceeds in 2 phases, training and testing as follow:-


Training Phase:
===============
```
clf = CAC(n_clusters, alpha, classifier=CLASSIFIER)
clf.fit(X_train, y_train)
```

## Input:
- X_train: The input data which is a normalized numpy matrix.
- y_train: The binary labels of input data points
- alpha: Hyperparameter
- classifier: The choice of base classifier. Choose from
	- LR: Logistic Regression (default)
	- RF: Random Forest with 10 estimators
	- SVM: Linear SVM
	- Perceptron: Linear Perceptron
	- DT: Decision Tree
	- Ridge: Ridge Classifier
	- SGD: Stochastic Gradient Descent classifier
	- LDA: Fischer's LDA classifier
	- KNN: k-Nearest Neighbour (k=5)
	- NB: Naive Bayes Classifier


## Output:
- clusters: the cluster centroids (µ+, µ- and µ) for all clusters and their F1/RoC values
- models: The trained models
- alt_labels: the labels for all testing points in all iterations
- errors: The unsupervised loss of CAC for every iteration
- seps: The sum of inter-class distances of all clusters for every iteration
- loss: The supervised loss of the models trained in every iteration


Testing/Evaluation Phase:
=========================

```
y_pred, y_proba = c.predict(X_test, -1) # get the predictions at the last (-1) iteration
f1 = f1_score(y_pred, y_test)
auc = roc_auc_score(y_test, y_proba)
```

## Input:
- X_test: The testing data
- y_test: The labels of test data. Used to compare with the test predictions to obtain the scores

## Output:
- test_scores: An array of arrays containing Accuracy, F1, AUC, Specificity and Sensitivity scores for every iteration. The first value in every array has the performance metric for classifiers trained on the input clusters given by the user.
