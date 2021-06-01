# ------------------------ EC3 code ----------------------------------#
# based on paper by Dr. Tanmoy chakraborty 

#by Mayank & pratham 

#==============================================================================
# #                           load the  basic libraries                          #
#==============================================================================
import numpy as np
#import scipy 
import pandas as pd
from copy import copy
import time
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import datasets

#==============================================================================
# #                       Load ML libraries                                       #
#==============================================================================
#Import Library
import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.preprocessing import StandardScaler


#load the libraries 
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
import argparse

#==============================================================================
#                            code starts                             #
#==============================================================================
# user defined function to change categories to number 

#==============================================================================
# #--------           TRAIN TEST SPLIT DONE --------------#
#==============================================================================

def run(X, train, test):
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
    scale = StandardScaler()
    X_train = scale.fit_transform(X_train)
    X_test = scale.fit_transform(X_test)

    O = len(X_test)
    L = len(np.unique(y))

    def cattoval(coln):
        for i in range(len(coln)):
            for j in range(L):
                if(coln[i] == cat[j]):
                    coln[i] = j
        return coln

    cat = []
    cat = list(np.unique(y))


    #==============================================================================
    # #                       SUPERVISED MACHINE LEARNING ALGORITHM START         #
    #==============================================================================

    Salgo = np.zeros(shape=(O, 0), dtype = np.int64) 
    grp = []
    accuracy = []
    names = []
    creport=[]
    models = []
    models.append(('DT - gini ', tree.DecisionTreeClassifier(criterion='gini')))
    models.append(('NB', GaussianNB()))
    models.append(('KNN', KNeighborsClassifier(n_neighbors=5)))
    models.append(('RF', RandomForestClassifier()))
    models.append(('SGD', SGDClassifier(loss="hinge", penalty="l2")))
    # models.append(('NN ', MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)))
    models.append(('LR ', LogisticRegression(class_weight='balanced', random_state=0, max_iter=1000)))
    models.append(('GBM',GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)))
    models.append(('SVM',LinearSVC()))
    models.append(('ADB',AdaBoostClassifier(n_estimators=100)))

    # # print("What classification algorithms you would like to choose ?")

    # n = int(input("Please the enter the corresponding number of classification algorithms"))
    n = 6
    # print("Decision Tree :- 1")
    # print("Naive Bayes :- 2")
    # print("KNN :- 3")
    # # print("Random Forest Tree :- 4")
    # print("SGD :- 5")
    # # print("Neural Network :- 6")
    # print("LR :- 6")
    # # print("Gradient Boost :- 7")
    # print("Support vector machine :- 8")
    # # print("Adaboost :- 9")

    a = [1, 2, 3, 5, 6, 8]
    # while len(a) < n:
    #     item = int(input("Enter your algorithm to the list: "))
    #     a.append(item)

    # # print("Here are your chosen algorithms")
    # print(a)

    def classifier(a, Salgo, names, accuracy, grp):
        # global Salgo
        # global accuracy
        # global names
        # global grp
        
        for i in a:
            print("Training Model ", i)
            if ( i ==5):
                i = i-1
                model = models[i][1]
                model.fit(X_train,y_train.ravel())
                predicted = model.predict(X_test)
                acc = accuracy_score(y_test,predicted)
                predicted = cattoval(predicted)
                Salgo = np.c_[Salgo,predicted]
                names.append(models[i][0])
                accuracy.append(acc)
                grp.append(L)
            else:
                i = i-1
                model = models[i][1]
                model.fit(X_train, y_train.ravel())
                model.score(X_train, y_train)
                predicted= model.predict(X_test)
                acc = accuracy_score(y_test,predicted)
                predicted = cattoval(predicted)
                Salgo = np.c_[Salgo,predicted]
                names.append(models[i][0])
                accuracy.append(acc)
                grp.append(L)

        return Salgo, names, accuracy, grp
                
    Salgo, names, accuracy, grp = classifier(a, Salgo, names, accuracy, grp)
    Salgo = np.array(Salgo,dtype = np.int64)


    #==============================================================================
    #--------------------       UNSUPERVISED MACHINE LEARNING ALGO      -------------------#
    #==============================================================================

    Ualgo = np.zeros(shape=(O, 0),dtype = np.int64) 

    Umodels = []
    Umodels.append(('KMeans', KMeans(n_clusters=L, random_state=0)))
    Umodels.append(('Affinity', AffinityPropagation(preference=-50)))
    Umodels.append(('DBSCAN', DBSCAN(eps=0.3, min_samples=10)))
    Umodels.append(('MeanShift', MeanShift( bin_seeding=True)))
    Umodels.append(('Hierarchical', AgglomerativeClustering()))

    # # print("What clustering algorithms you would like to choose ?")

    # n = int(input("Please the enter the corresponding numbers to apply the algorithms"))
    n = 5
    # print("Kmeans :- 1")
    # print("Affinity propagation :- 2")
    # print("DBSCAN :- 3")
    # print("Mean Shift :- 4")
    # print("Agglomerative  :- 5")

    b = [1, 3]
    # while len(b) < n:
    #     item = int(input("Enter your algorithm to the list: "))
    #     b.append(item)

    # print("Here are your chosen algorithms")
    # print(b)

    def clustering(b, Ualgo, accuracy, names, grp):
        # global Ualgo
        # global accuracy
        # global names
        # global grp

        for i in b:
            print("Clustering Algo ", i)
            i = i-1
            model = Umodels[i][1]
            model.fit(X)
            labels = model.labels_
            index_test = test
            predicted = [labels[j] for j in index_test]
            n_clusters_ = len(np.unique(predicted))
            predicted = [((n_clusters_ ) - 1)if x == -1 else x for x in predicted]
            Ualgo = np.c_[Ualgo,predicted]
            names.append(Umodels[i][0])
            grp.append(n_clusters_)
            accuracy.append(0)
        return Ualgo, accuracy, names, grp

    Ualgo, accuracy, names, grp = clustering(b, Ualgo, accuracy, names, grp)

    summary = pd.DataFrame(np.column_stack([names,accuracy,grp]), columns=["algorithm","accuracy","Grp"])
     
    #==============================================================================
    # # --------------------    Parameter initialisation  ---------------------------------#
    #==============================================================================

    N = len(X_test)                        #  defined
    # L                            number of classes
    C1= len(a)                    #no. of classifier
    C2 = len(b)                  # no. of clusters
    C = C1 + C2
    G1 = sum(grp[:len(a)])
    G2 = sum(grp[len(a):len(grp)])
    G = G1+G2

    Algo = np.column_stack((Salgo, Ualgo))
    #==============================================================================
    # # --------------------------- Matrix Formation ------------------------------------------------#
    #==============================================================================

    #------------------------------------------------------------------------------------------#
    # s = time.time()
    # print ("starting building matrices")

    def normalise(A,n=1):  
        if n == 1 :
            A = A/A.sum(axis=1)[:,None]
        elif n == 0:
            A = A/A.sum(axis=0)[None,:]
        return A

    #-------------------------------------------------------------------------------------#
    #-------------K from A -------------# 
    from numpy import linalg as LA
    from math import sqrt
    from copy import copy, deepcopy

    def StochasticK(Ac,ep = 0.001):
        K = np.zeros((X_test, X_test))
        KM =copy(Ac)
        N = len(X_test)
    #    KM = np.zero((int(Ac.shape[0]), int(Ac.shape[1])))

        while LA.norm(np.array(K)-np.array(KM)) / (N*N) > eps:
            K=deepcopy(KM)
    #        d=KM.sum(axis=1)
    #         for i in range(N):
    #             for j in  range(N):
    #                 K[i][j] = KM[i][j] / d[i] 
            K = normalise(KM,1)
            for i in range(N): 
                for j in  range(N): #52
                    K[i][j] = K[j][i]= sqrt(K[i][j]*K[j][i])
        return K 
    #-----------------------------------------------------------------------------------#

    #==============================================================================
    # #-----------------------------------     MEMBERSHIP MATRIX ------------------------------------#
    #==============================================================================
    def getMemMat():
        MemMat = np.zeros((N, G))
        index = 0
        for k in range(C):
            for i in range(N):
                j = index +  Algo[i,k]
                if j < G:
                    MemMat[i][j] = 1
            index = index + grp[k]
        return MemMat


    MemMat = getMemMat()                      #  NxG
    MemDF = pd.DataFrame(MemMat)        
    #MemMat = normalise(MemMat,0)


    #==============================================================================
    # #-----------              Co-occurence Matrix                  -----------#
    #==============================================================================

    def Count(m,n):
        score = 0
        for k in range(C):    
            if ( Algo[m,k] == Algo[n,k]):
                score = score + 1
        return score
        
    CoMat = np.zeros((N, N))        # N x N

    for i in range(N):
        for j in range(i,N):
            value = Count(i,j)
            CoMat[i][j]=CoMat[j][i] = value

    CoMat = normalise(CoMat)
    #CoMat = StochasticK(CoMat)

    #--------------------  average object class matrix -------------------------------#
    #--------------------only supervised algo is considered-------------------------#


    def fun(m,n):
        score = 0
        for k in range(C1):
            if Algo[m,k] == n:
                score = score + 1
        return score

    def getObjclass():
        Objclass = np.zeros((N, L))
        for i in range(N):
            for j in  range(L):
        #        temp = int (Salgo[i,j])
                value = fun(i,j)
                value2 = value/float(4)
        #        Objclass[i][temp]= Objclass[i][temp] + 1/4
                Objclass[i,j] = value2
        return Objclass
            
    Objclass = getObjclass()

    #----------------- average group class matrix --------------------------------#
    def getGrpclass():
    #    global MemDF
        Grpclass = np.zeros(shape=(G,L))
        for gno in range(G):
            idx = MemDF[MemDF[gno] == 1 ].index.tolist()
            tot = float(len(idx) * C1)
            for i in idx:
                for j in range(C1):
                    score = Salgo[i,j] 
                    Grpclass[gno,score]= (Grpclass[gno,score] + 1/tot )   # for average
        return Grpclass

    Grpclass = getGrpclass()


    #------------------- object - class matrix ------------------------------------#
    """
    # condition satisfied   Fo >= 0 , |Fo i. | = 1  for every i in 1:n
    Fo = np.zeros(shape=(len(X_test),13))

    for i in range (len(X_test)):
        Fo[i][0] = 1
        
    """
    Fo = np.random.rand(N,L)

    Fo = Fo/Fo.sum(axis=1)[:,None]




        
    #------------------- Group - class matrix --------------------------------------#
    # condition satisfied   Fg >= 0 , |Fg .j | = 1  for every j in 1:l

    Fg = np.random.rand(G,L)

    Fg = Fg/Fg.sum(axis=0)[None,:]




    """
    Fg = np.zeros(shape=(13*5,13))

    for i in range ((int(Fg.shape[1]))):
        Fg[0][i] = 1
    """
    # e = time.time()
    # print ("all matrices have been made ")
    # # print (e-s)

    #-------------------------------------------------------------------------------------------------------#

    #==============================================================================
    # # ---------------- All Matrix Built --------------------------------------------#
    #==============================================================================

    # import time 
    from numpy import linalg as LA
    from math import sqrt
    from copy import copy, deepcopy
    # input Km Kc Yo Yg alpha beta gamma delta epsilon
    # initialised Fo & Fg with condition preserved 
    # output Fo ( N x l ) probability of each element N belonging to class l 
    def getdiagonal (A, x):
        A = A.sum(axis = x)
        return np.diag(A)


    Dm = getdiagonal(MemMat , 0)
    one = np.ones((G,G))
    Dmdash = getdiagonal(MemMat,1)
    Dc     = getdiagonal(CoMat,0)
    oneN = np.ones((N,N))
    ideN = np.identity(N)

        
    def EC3(Fo , Fg , Km , Kc , Yo , Yg , alpha = 0.25 , beta =0.35, gamma = 0.35, delta = 0.35 , eps = 0.0001):
        t =1
        Fot = copy(Objclass)
        
        while LA.norm(np.array(Fot)-np.array(Fo)) / (N*L) > eps :             #Fo = Fo(t-1)
           # global t
           # t = t + 1 
            # print ("loop run")
            lhs = np.linalg.inv(2*delta*one + alpha * Dm)                     #GxG
            rhs = alpha * np.matmul(Km.transpose() , Fo) + 2*delta*Yg         #GxL
            Fg =  np.matmul(lhs,rhs)                                          # GxG x GxL = G x L
            
            a = alpha *Dmdash
            b = 2*beta*Dc
            c = beta* np.matmul(ideN, Kc)
            d= beta * np.matmul(oneN,Kc)
            e = 2*gamma*oneN
           
            
            s = a + b
             
            s = s - c
            s = s - d
            s = s + e
            lhs = np.linalg.inv(s)
            f = alpha * np.matmul(Km,Fg)
            g  = 2*gamma*Yo
            rhs = f + g
            
            Fo = copy(Fot)
            Fot = np.matmul(lhs,rhs)
            
            # # print (Fot)
        return Fot
      
    # startAlgo = time.time() 

    MainMat = EC3(Fo,Fg,MemMat,CoMat,Objclass,Grpclass,0.25,0.35,0.35,0.05,0.0001)
        
    # Endalgo = time.time()
    # # print  (Endalgo - startAlgo)
    output = pd.DataFrame(MainMat)


    output.to_csv("outputmatrix.csv")

    result = np.argmax(MainMat, axis=1)
    #y_test = cattoval(y_test)
    # print("Calulating Accuracy")
    accu = accuracy_score(y_test,result)
    f1 = f1_score(y_test,result)
    roc = roc_auc_score(y_test,MainMat[:,1])

    print("f1 = ", f1)
    print("roc = ", roc)
    print("\n")
    return f1

datasets = ["adult", "cic", "creditcard",\
            "magic", "sepsis", "titanic"]

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ALL')
parser.add_argument('--cv', default='False')
args = parser.parse_args()  

if args.dataset == "ALL":
    data = datasets
else:
    data = [args.dataset]

# DATASET = "adult" # see folder, *the Titanic dataset is different*
for DATASET in data:
    print(DATASET)
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

        train = range(len(X_train))
        test = range(len(X_train), len(X))
        # for i in range(5):
        #     run(X, train, test)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=108)
        f1 = []
        if args.cv == "True":
            for train, test in skf.split(X, y):
                print("#### Performing Stratified k-Fold ####")
                f1.append(run(X, train, test))
            print("Avg. F1: ", np.mean(f1))
        else:
            for i in range(5):
                all_indices = range(len(X))
                train, test = train_test_split(all_indices, stratify=y, random_state=108)
                f1.append(run(X, train, test))
            print("Avg. F1: ", np.mean(f1), "(", np.std(f1), ")")


    elif DATASET == "titanic":
        X_train = pd.read_csv("./data/" + DATASET + "/" + "X_train.csv", header=None).to_numpy()
        X_test = pd.read_csv("./data/" + DATASET + "/" + "X_test.csv", header=None).to_numpy()
        y_train = pd.read_csv("./data/" + DATASET + "/" + "y_train.csv", header=None).to_numpy()
        y_test = pd.read_csv("./data/" + DATASET + "/" + "y_test.csv", header=None).to_numpy()

        X = np.vstack([X_train, X_test])
        y = np.vstack([y_train, y_test])

        train = range(len(X_train))
        test = range(len(X_train), len(X))
        f1 = []
        # for i in range(5):
        #     run(X, train, test)
        if args.cv == "True":
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=108)
            for train, test in skf.split(X, y):
                print("#### Performing Stratified k-Fold ####")
                f1.append(run(X, train, test))
            print("Avg. F1: ", np.mean(f1), "(", np.std(f1), ")")
        else:
            for i in range(5):
                all_indices = range(len(X))
                train, test = train_test_split(all_indices, stratify=y, random_state=108)
                f1.append(run(X, train, test))
            print("Avg. F1: ", np.mean(f1), "(", np.std(f1), ")")


    ###########################################

    else:
        X = pd.read_csv("./data/" + DATASET + "/" + "X.csv").to_numpy()
        y = pd.read_csv("./data/" + DATASET + "/" + "y.csv").to_numpy()
        f1 = []
        if args.cv == "True":
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=108)
            for train, test in skf.split(X, y):
                print("#### Performing Stratified k-Fold ####")
                f1.append(run(X, train, test))
            print("Avg. F1: ", np.mean(f1))
        else:
            for i in range(5):
                all_indices = range(len(X))
                train, test = train_test_split(all_indices, stratify=y, random_state=108)
                f1.append(run(X, train, test))
            print("Avg. F1: ", np.mean(f1), "(", np.std(f1), ")")