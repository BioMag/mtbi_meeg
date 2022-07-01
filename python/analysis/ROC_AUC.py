# Copyright by Verna Heikkinen 2022
# 
# Scrtpt for plotting ROC and computing AUC for mTBI classification results
#
import os
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay, auc
from sklearn.covariance import OAS
from TBIMEG import load_subject, load_group, load_dataset, LOOCV_manager, CV_foldmanager


def CV_solver(solver, X, y):
    
    folds = CV_foldmanager(y, cv_folds=10)
    
    tprs = [] #save results for plotting
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    for train_ids, test_ids in folds:
    
        if solver == "LDAoas":
            oa = OAS(store_precision=False, assume_centered=False)
            clf = LinearDiscriminantAnalysis(solver='lsqr', covariance_estimator=oa,
                                            priors = np.array([0.5, 0.5]))
        elif solver == "SVM":
            clf = SVC(kernel="linear", C=0.025)
        elif solver == "LR":
            clf = LogisticRegression(max_iter=2000)   
        else:
            raise("muumi")

        X_train = X[train_ids]
        X_test = X[test_ids]

        y_train = y[train_ids]
        y_test = y[test_ids]

        clf.fit(X_train, y_train)
        pred = clf.predict(X_test).astype(int)
        
        viz = RocCurveDisplay.from_estimator(
        clf,
        X[test_ids],
        y[test_ids],
        name="",
        alpha=0.3,
        lw=1,
        ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    return test_ids, pred, tprs, aucs, mean_fpr


def ROC_results(LOOCV_results):
    
    tprs, aucs = LOOCV_results[2], LOOCV_results[3]
    mean_fpr = LOOCV_results[4]
    

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title="10-fold CV ROC curve",
    )
    #ax.legend(loc="lower right") #clutters the image

    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #parser.add_argument('--threads', type=int, help="Number of threads, using multiprocessing", default=1) #skipped for now
    parser.add_argument('--iter', type=int, help="Current iteration") #not sure I want to do 50 iterations.....

    parser.add_argument('--band', type=str, help="Band")
    parser.add_argument('--drop', type=str, help="Band_only or Band_drop")
    parser.add_argument('--sensor', type=str, help="sensor: mag, grad, both")
    parser.add_argument('--eyes', type=str, help="eo or ec")
    parser.add_argument('--location', type=str, help="Otaniemi or BioMag")
    parser.add_argument('--clf', type=str, help="classifier", default="LDAoas")

    args = parser.parse_args()
    
    if args.location == "Otaniemi":
        LOC = "ON"
    else:
        LOC = "BM"
    
    #selection_file = f"{args.location}_select_{args.eyes}.csv"
    selection_file = f"DATA/{args.location}_select.csv"
    df = pd.read_csv(selection_file, index_col = 0)
    file_selection = df.loc[f"pred_{args.iter}"]

    save_folder = f"RESULTS/{args.drop}/{args.location}/{args.clf}/{args.sensor}"
    save_file = f"{save_folder}/{LOC}_{args.band}_{args.clf}_{args.sensor}_{args.eyes}.csv"

    bands = [[0,7.6], [7.6,13],[13,30],[30,90], [0,0]]
    band_name = ["slow", "alpha", "beta", "high", "all"]
    band = bands[band_name.index(args.band)]

    # Read frequencies 
    freqs = np.loadtxt("f.txt", delimiter = ",")[:,0]

    freqs_sel = (freqs >= band[0]) & (freqs < band[1])
    if args.drop == "Band_only":
        freqs_sel = freqs_sel == False
        

    patient_path = f"DATA/{args.location}/patient"
    control_path = f"DATA/{args.location}/control"

    _index = ["case", "true"]
    #results = []

    print(f"TBIMEG classifcation data from {args.location} on {args.eyes} condition.")
    print(f"Using {args.sensor} sensors with {args.clf} classifier. Iteration {args.iter}")
    print("Load data")
    X, y, s, i = load_dataset(patient_path, control_path, 1, 2, args.eyes, args.sensor, freqs_sel, file_selection)
    X = preprocessing.scale(X)
    
    fig, ax = plt.subplots() #TODO: should these be given for the functions?

    results = CV_solver(args.clf, X, y)
    ROC_results(results)
    
    plt.savefig(fname=f"ROC_{LOC}_{args.band}_{args.clf}_{args.sensor}_{args.eyes}.pdf")
