# import libraries
from __future__ import division, print_function

import argparse

import os
import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.covariance import OAS

from joblib import Parallel, delayed
import multiprocessing

from datetime import datetime


def load_subject(group_path, subject, eyes, channel_type, freqs_sel,index):
    
    if channel_type == "mag":
        channels = list(np.arange(0,306, 3))
    elif channel_type == "grad":
        channels = list(np.arange(1,306, 3))
        channels += list(np.arange(2,306, 3))
    else:
        channels = list(np.arange(0,306))
    

    filename = f"{group_path}/{subject}/{subject}_spectra_{eyes}_{index}.npy"
    
    arr = np.load(filename)
    arr = arr[channels]
    arr = arr[:, freqs_sel]
    arr = np.reshape(arr, -1)
        
    return arr, index


def load_group(group_path, eyes, group_code, channel_type, freqs_sel, selection):
    
    X = []
    y = []

    subjects = [s for s in os.listdir(group_path) if s[0] != "."]
    subjects.sort()
    
    subjs = []
    indicies = []
    for i, s in enumerate(subjects):
        
        index = selection[str(s)]
        _X, ind = load_subject(group_path, s, eyes, channel_type, freqs_sel, index)
        
        X.append(_X)
        y.append(group_code)
        
        subjs.append(s)
        indicies.append(ind)
        
    return X, y, subjs, indicies
        

def load_dataset(patient_path, control_path, patient_code, control_code, 
                 eyes, channel_type, freqs_sel, selection):
    
    
    X_p, y_p, s_p, i_p = load_group(patient_path, eyes, patient_code, channel_type, freqs_sel,selection)
    X_c, y_c, s_c, i_c = load_group(control_path, eyes, control_code, channel_type, freqs_sel,selection)
    
    X = X_p + X_c
    y = y_p + y_c
    
    s = s_p + s_c
    i = i_p + i_c
    
    X = np.array(X)
    y = np.array(y)

    
    return X, y, s, i


def LOOCV_manager(y):
    
    n = len(y)

    folds = []

    for cv in range(n):
        
        test_ids = np.arange(cv, n, n)
        train_ids = np.array([i for i in range(n) if i not in list(test_ids)])

        fold = [train_ids, test_ids]
        folds.append(fold)
    
    return folds


def CV_foldmanager(y, cv_folds):
    
    n = len(y)

    folds = []

    for cv in range(cv_folds):
        
        test_ids = np.arange(cv, n, cv_folds)
        train_ids = np.array([i for i in range(n) if i not in list(test_ids)])

        fold = [train_ids, test_ids]
        folds.append(fold)
    
    return folds

def CV_parallel(fold, solver, X, y):
    
    train_ids = fold[0]
    test_ids = fold[1]
    
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
    
    return test_ids, pred

def performCV_parallel(folds, X, y, solver, n_jobs):

    N_samples = len(y)

    y_pred = np.empty(N_samples)
    
    results = Parallel(n_jobs = n_jobs)(delayed(CV_parallel)(fold, solver, X, y) for fold in folds)

    for ids, pred in results:
    
        y_pred[ids] = pred
        
    return y_pred



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--threads', type=int, help="Number of threads, using multiprocessing", default=1)
    parser.add_argument('--iter', type=int, help="Current iteration")

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
    
    selection_file = f"{args.location}_selection_{args.eyes}.csv"
    df = pd.read_csv(selection_file, index_col = 0)
    file_selection = df.loc[f"pred{args.iter}"]

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
    results = []

    print(f"TBIMEG classifcation data from {args.location} on {args.eyes} condition.")
    print(f"Using {args.sensor} sensors with {args.solver} classifier. Iteration {args.iter}")
    print("Load data")
    X, y, s, i = load_dataset(patient_path, control_path, 1, 2, args.eyes, args.sensor, freqs_sel, file_selection)
    X = preprocessing.scale(X)

    print(f"Print X.shape: {X.shape}")
                
    results.append(s)
    results.append(y)
    
    now = datetime.now()

    print("Classify")
    now = datetime.now()
    print(f" {now}")
    folds = LOOCV_manager(y)
    y_pred = performCV_parallel(folds, X, y, args.clf, args.threads).astype(int)

    results.append(y_pred)    
    _index.append(f"pred{args.iter}")

    dfr = pd.DataFrame(results, index = _index)
    dfr.to_csv(save_file)
    now = datetime.now()
    print("Saved")
    print(f" {now}")
