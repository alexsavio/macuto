#!/usr/bin/python

import os
import re
import sys
import argparse
import numpy as np
import nibabel as nib
import pickle

import scipy.stats as stats

#data preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#classification
from sklearn import tree
from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.mixture import GMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron

#feature selection
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectFdr
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import RFECV
from sklearn.cross_validation import KFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import zero_one

#other decompositions
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
from sklearn.lda import LDA

#pipeline
from sklearn.cross_validation import KFold
from sklearn.cross_validation import LeaveOneOut
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion

#scores
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import classification_report

#other decompositions
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
from sklearn.lda import LDA
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE

#pipelining
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import LeaveOneOut, StratifiedKFold
from sklearn.pipeline import Pipeline, FeatureUnion

#debugging
from IPython.core.debugger import Tracer; debug_here = Tracer()

sys.path.append('/home/alexandre/Dropbox/Documents/phd/work/aizkolari')
import aizkolari_utils as au

#bash
'''
#OUTPUT DIRECTORY
d='/home/alexandre/Dropbox/Documents/phd/work/oasis_feets'

#CLASSIFIERS
es="linsvm svm cart rf"

#DATA SETTINGS
wd='/home/alexandre/Dropbox/Documents/phd/work/oasis_feets'
dataf=$wd/oasis_warpfield_hist3d.npy
labsf=$wd/oasis_warpfield_hist3d_labels.txt
#dataf=os.path.join(wd, 'oasis_warpfield_hist3d.npy')
#labelsf=os.path.join(wd, 'oasis_warpfield_hist3d_labels.txt')

n_cpus=2

#cv-folding
#cvfold=10
cvfold=loo

for e in $es; do
    echo $e
    ${wd}/do_classification.py -d $dataf -l $labsf --fsmethod none -e $e -c ${n_cpus} --cvfold ${cvfold}
done;
'''

#python
'''
wd='/home/alexandre/Dropbox/Documents/phd/work/oasis_feets'

dataf=os.path.join(wd, 'oasis_warpfield_hist3d.npy')
labelsf=os.path.join(wd, 'oasis_warpfield_hist3d_labels.txt')

n_cpus = 2

clfmethod = 'linsvm'
fsname    = 'none'

cvfold = 'cv'
n_folds = 10

'''

#-------------------------------------------------------------------------------
def set_parser():

    clfmethods   = ['cart', 'gmm', 'rf', 'svm', 'sgd', 'linsvm', 'percep']
    prefsmethods = ['none', 'pearson', 'bhattacharyya', 'welcht']
    fsmethods    = ['none', 'stats', 'rfe', 'rfecv', 'univariate', 'fdr', 'fpr', 'extratrees', 'pca', 'rpca', 'lda'] #svmweights

    parser = argparse.ArgumentParser(description='OASIS AAL classification experiment.')
    parser.add_argument('-d', '--dataf',   dest='dataf',     default='', required=True, help='data npy file with a matrix NxD, where N is n_subjs and D n_feats')
    parser.add_argument('-l', '--labelsf',  dest='labelsf',    default='', required=True,   help='list file with labels for each subject in the same order as in the data file.')
    parser.add_argument('-o', '--outdir',    dest='outdir',      default='', required=False,  help='output data directory path. Will use datadir if not set.')
    parser.add_argument('--prefsmethod',     dest='prefsmethod', default='none', choices=prefsmethods, required=False, help='Pre-feature selection method')
    parser.add_argument('--prefsthr',        dest='prefsthr',    default=95, type=int, required=False, help='Pre-feature selection method threshold [0-100]')
    parser.add_argument('--fsmethod',        dest='fsmethod',    default='stats', choices=fsmethods, required=True, help='feature extraction method used to build the datasets')
    parser.add_argument('--cvfold',          dest='cvfold',      default='10', choices=['10', 'loo', 'cv'], required=False, help='Cross-validation folding method: stratified 10-fold, leave-one-out or any other K for a stratified K-fold cv (specify it with --nfolds).')
    parser.add_argument('-k', '--nfolds',    dest='nfolds',       default=5, required=False, type=int, help='number of folds in case cv is used in --cvfold.')
    parser.add_argument('-e', '--estim',     dest='estimator',   default='svm', choices=clfmethods, required=False, help='classifier type')
    parser.add_argument('-c', '--ncpus',     dest='ncpus',       default=1, required=False, type=int, help='number of cpus used for parallelized grid search')
    parser.add_argument('-v', '--verbosity', dest='verbosity',   default=2, required=False, type=int, help='Verbosity level: Integer where 0 for Errors, 1 for Input/Output, 2 for Progression reports')

    return parser


#-------------------------------------------------------------------------------
def list_filter (list, filter):
    return [ (l) for l in list if filter(l) ]

#-------------------------------------------------------------------------------
def dir_search (regex, wd='.'):
    ls = os.listdir(wd)

    filt = re.compile(regex).search
    return list_filter(ls, filt)

#-------------------------------------------------------------------------------
def dir_match (regex, wd='.'):
    ls = os.listdir(wd)

    filt = re.compile(regex).match
    return list_filter(ls, filt)

#-------------------------------------------------------------------------------
def list_match (regex, list):
    filt = re.compile(regex).match
    return list_filter(list, filt)

#-------------------------------------------------------------------------------
def list_search (regex, list):
    filt = re.compile(regex).search
    return list_filter(list, filt)

#-------------------------------------------------------------------------------
def pre_featsel (X, y, method, thr=95):

    #pre feature selection, measuring distances
    #Pearson correlation
    if method == 'pearson':
        au.log.info ('Calculating Pearson correlation')
        m = np.abs(pearson_correlation (X, y))

    #Bhattacharyya distance
    elif method == 'bhattacharyya':
        au.log.info ('Calculating Bhattacharyya distance')
        m = bhattacharyya_dist (X, y)

    #Welch's t-test
    elif method == 'welcht':
        au.log.info ("Calculating Welch's t-test")
        m = welch_ttest (X, y)

    #threshold data
    if method != 'none':
        mt = au.threshold_robust_range (m, thr)

    return mt

#-------------------------------------------------------------------------------
def pearson_correlation (X, y):

    #number of features
    n_feats = X.shape[1]

    #creating output volume file
    p = np.zeros(n_feats)

    #calculating pearson accross all subjects
    for i in range(X.shape[1]):
      p[i] = stats.pearsonr (X[:,i], y)[0]

    p[np.isnan(p)] = 0

    return p


#-------------------------------------------------------------------------------
def bhattacharyya_dist (X, y):

    classes = np.unique(y)
    n_class = len(classes)
    n_feats = X.shape[1]

    b = np.zeros(n_feats)
    for i in np.arange(n_class):
        for j in np.arange(i+1, n_class):
            if j > i:
                xi = X[y == i, :]
                xj = X[y == j, :]

                mi = np.mean (xi, axis=0)
                mj = np.mean (xj, axis=0)

                vi = np.var  (xi, axis=0)
                vj = np.var  (xj, axis=0)

                si = np.std  (xi, axis=0)
                sj = np.std  (xj, axis=0)

                d  = 0.25 * (np.square(mi - mj) / (vi + vj)) + 0.5  * (np.log((vi + vj) / (2*si*sj)))
                d[np.isnan(d)] = 0
                d[np.isinf(d)] = 0

                b = np.maximum(b, d)

    return b

#-------------------------------------------------------------------------------
def welch_ttest (X, y):

    classes = np.unique(y)
    n_class = len(classes)
    n_feats = X.shape[1]

    b = np.zeros(n_feats)
    for i in np.arange(n_class):
        for j in np.arange(i+1, n_class):
            if j > i:
                xi = X[y == i, :]
                xj = X[y == j, :]
                yi = y[y == i]
                yj = y[y == j]

                mi = np.mean (xi, axis=0)
                mj = np.mean (xj, axis=0)

                vi = np.var  (xi, axis=0)
                vj = np.var  (xj, axis=0)

                n_subjsi = len(yi)
                n_subjsj = len(yj)

                t = (mi - mj) / np.sqrt((np.square(vi) / n_subjsi) + (np.square(vj) / n_subjsj))
                t[np.isnan(t)] = 0
                t[np.isinf(t)] = 0

                b = np.maximum(b, t)

    return b

#-------------------------------------------------------------------------------
def classification_metrics (targets, preds, probs):
    fpr, tpr, thresholds = roc_curve(targets, probs[:, 1])
    roc_auc = auc(fpr, tpr)

    cm = confusion_matrix(targets, preds)

    #accuracy
    accuracy =accuracy_score(targets, preds)

    #recall? True Positive Rate or Sensitivity or Recall
    recall = recall_score(targets, preds)

    #precision
    precision = precision_score(targets, preds)

    tnr = 0.0
    #True Negative Rate or Specificity?
    if len(cm) == 2:
        tnr = float(cm[0,0])/(cm[0,0] + cm[0,1])

    out = {}
    out['accuracy'] = accuracy
    out['recall'] = recall
    out['precision'] = precision
    out['tnr'] = tnr
    out['roc_auc'] = roc_auc

    return out

#-------------------------------------------------------------------------------
def plot_roc_curve (targets, preds, probs):
    import pylab as pl
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(targets, probs[:, 1])
    roc_auc = auc(fpr, tpr)
    pl.plot(fpr, tpr, lw=1, label='ROC LOO-test (area = %0.2f)' % (roc_auc))
    pl.xlim([-0.05, 1.05])
    pl.ylim([-0.05, 1.05])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('ROC for ' + feats + ' ROI ' + roinom)
    pl.legend(loc="lower right")
    pl.show()

#-------------------------------------------------------------------------------
def get_clfmethod (clfmethod, n_feats, n_subjs, n_jobs=1):

    #classifiers
    classifiers = { 'cart'   : tree.DecisionTreeClassifier(random_state = 0),
                    'rf'     : RandomForestClassifier(max_depth=None, min_samples_split=1, random_state=None),
                    'gmm'    : GMM(init_params='wc', n_iter=20, random_state=0),
                    'svm'    : SVC (probability=True, max_iter=50000, class_weight='auto'),
                    'linsvm' : LinearSVC (class_weight='auto'),
                    'sgd'    : SGDClassifier (fit_intercept=True, class_weight='auto', shuffle=True, n_iter = np.ceil(10**6 / 416)),
                    'percep' : Perceptron (class_weight='auto'),
    }

    #Classifiers parameter values for grid search
    if n_feats < 10:
        max_feats = range(1, n_feats, 2)
    else:
        max_feats = range(1, 30, 4)
    max_feats.extend([None, 'auto', 'sqrt', 'log2'])

    clgrid =      { 'cart'   : dict(criterion = ['gini', 'entropy'], max_depth = [None, 10, 20, 30]),
                    'rf'     : dict(n_estimators = [3, 5, 10, 30, 50, 100], max_features = max_feats),
                    'gmm'    : dict(n_components = [2,3,4,5], covariance_type=['spherical', 'tied', 'diag'], thresh = [True, False] ),
                    #'svm'  : dict(kernel = ['rbf', 'linear', 'poly'], C = np.logspace(-3, 3, num=7, base=10), gamma = np.logspace(-3, 3, num=7, base=10), coef0 = np.logspace(-3, 3, num=7, base=10)),
                    #'svm'    : dict(kernel = ['rbf', 'poly'], C = np.logspace(-3, 3, num=7, base=10), gamma = np.logspace(-3, 3, num=7, base=10), coef0=np.logspace(-3, 3, num=7, base=10)),
                    'svm'    : dict(kernel = ['rbf', 'linear'], C = np.logspace(-3, 3, num=7, base=10), gamma = np.logspace(-3, 3, num=7, base=10)),
                    'linsvm' : dict(C = np.logspace(-3, 3, num=7, base=10)),
                    'sgd'    : dict(loss=['hinge', 'modified_huber', 'log'], penalty=["l1","l2","elasticnet"], alpha=np.logspace(-6, -1, num=6, base=10)),
                    'percep' : dict(penalty=[None, 'l2', 'l1', 'elasticnet'], alpha=np.logspace(-3, 3, num=7, base=10)),
    }

    return classifiers[clfmethod], clgrid[clfmethod]

#-------------------------------------------------------------------------------
def get_fsmethod (fsmethod, n_feats, n_subjs, n_jobs=1):

    if fsmethod == 'none':
        return '', None

    elif fsmethod == 'stats':
        return 'stats', None

    #Feature selection procedures
                                #http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
    fsmethods = { 'rfe'       : RFE(estimator=SVC(kernel="linear"), step=0.05, n_features_to_select=2),
                                #http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
                  'rfecv'     : RFECV(estimator=SVC(kernel="linear"), step=0.05, loss_func=zero_one), #cv=3, default; cv=StratifiedKFold(n_subjs, 3)
                                #Univariate Feature selection: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html
                  'univariate': SelectPercentile(f_classif, percentile=5),
                                #http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFpr.html
                  'fpr'       : SelectFpr (f_classif, alpha=0.05),
                                #http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFdr.html
                  'fdr'       : SelectFdr (f_classif, alpha=0.05),
                                #http://scikit-learn.org/stable/modules/feature_selection.html
                  'extratrees': ExtraTreesClassifier(n_estimators=50, max_features='auto', n_jobs=n_jobs, random_state=0),

                  'pca'       : PCA(n_components='mle'),
                  'rpca'      : RandomizedPCA(random_state=0),
                  'lda'       : LDA(),
    }

    #feature selection parameter values for grid search
    max_feats = ['auto']
    if n_feats < 10:
        feats_to_sel = range(2, n_feats, 2)
        n_comps = range(1, n_feats, 2)
    else:
        feats_to_sel = range(2, 20, 4)
        n_comps = range(1, 30, 4)
    max_feats.extend(feats_to_sel)

    n_comps_pca = list(n_comps)
    n_comps_pca.extend(['mle'])

    fsgrid =    { 'rfe'       : dict(estimator_params = [dict(C=0.1), dict(C=1), dict(C=10)], n_features_to_select = feats_to_sel),
                  'rfecv'     : dict(estimator_params = [dict(C=0.1), dict(C=1), dict(C=10)]),
                  'univariate': dict(percentile = [1, 3, 5, 10]),
                  'fpr'       : dict(alpha = [1, 3, 5, 10]),
                  'fdr'       : dict(alpha = [1, 3, 5, 10]),
                  'extratrees': dict(n_estimators = [1, 3, 5, 10, 30, 50], max_features = max_feats),
                  'pca'       : dict(n_components = n_comps_pca, whiten = [True, False]),
                  'rpca'      : dict(n_components = n_comps, iterated_power = [3, 4, 5], whiten = [True, False]),
                  'lda'       : dict(n_components = n_comps)
    }

    return fsmethods[fsmethod], fsgrid[fsmethod]


#-------------------------------------------------------------------------------
def shelve_vars (ofname, varlist):
   mashelf = shelve.open(ofname, 'n')

   for key in varlist:
      try:
         mashelf[key] = globals()[key]
      except:
         au.log.error('ERROR shelving: {0}'.format(key))

   mashelf.close()

#-------------------------------------------------------------------------------
def append_to_keys (mydict, preffix):
    return {preffix + str(key) : (transform(value) if isinstance(value, dict) else value) for key, value in mydict.items()}

#-------------------------------------------------------------------------------

def main(argv=None):

    parser  = set_parser()

    try:
       args = parser.parse_args ()
    except argparse.ArgumentError, exc:
       print (exc.message + '\n' + exc.argument)
       parser.error(str(msg))
       return -1

    labelsf     = args.labelsf.strip()
    dataf       = args.dataf.strip()
    outdir      = args.outdir.strip()
    clfmethod   = args.estimator.strip()
    fsname      = args.fsmethod.strip()
    prefsmethod = args.prefsmethod.strip()
    prefsthr    = args.prefsthr

    cvfold      = args.cvfold.strip()
    n_folds     = args.nfolds
    n_cpus      = args.ncpus
    verbose     = args.verbosity

    au.setup_logger(verbose, logfname=None)

    scale = True

    #label values
    n_class = 2

    #labels
    y = np.loadtxt(labelsf).astype(int)
    n_subjs = len(y)

    #data
    data    = np.load(dataf)
    n_feats = data.shape[1]

    if n_subjs != data.shape[0]:
        print ("Labels and data shapes do not coincide.")
        exit(-1)

    #results
    results = {}
    results['y'] = y

    classif, clp = get_clfmethod (clfmethod, n_feats, n_subjs, n_cpus)

    #feature selection method instance
    fsmethod, fsp = get_fsmethod (fsname, n_feats, n_subjs, n_cpus)

    #results variables
    preds   = {}
    truth   = {}
    rscore  = {} #np.zeros(n_subjs) #ROI weights, based on AUC
    f1score = {} #np.zeros(n_subjs) #ROI weights, based on F1-score
    probs   = {} #np.zeros((n_subjs, n_class))
    best_p  = {}

    #cross validation
    if cvfold == '10':
        ct = StratifiedKFold(y, 10)
    elif cvfold == 'loo':
        ct = LeaveOneOut(len(y))
    elif cvfold == 'cv':
        ct = StratifiedKFold(y, n_folds)

    fc = 0
    for train, test in ct:
        print '.',

        #train and test sets
        try:
            X_train, X_test, y_train, y_test = data[train,:], data[test,:], y[train], y[test]

        except:
            debug_here()

        #scaling
        if clfmethod == 'svm' or clfmethod == 'linsvm' or clfmethod == 'sgd':
            #scale_min = -1
            #scale_max = 1
            #[X_train, dmin, dmax] = au.rescale (X_train, scale_min, scale_max)
            #[X_test,  emin, emax] = au.rescale (X_test,  scale_min, scale_max, dmin, dmax)
            scaler  = MinMaxScaler((-1,1))
            #scaler  = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

        #classifier instance
        elif clfmethod == 'gmm':
            classif.means_ = np.array([X_train[y_train == i].mean(axis=0)
                             for i in xrange(n_class)])

        #PRE feature selection
        if prefsmethod != 'none':
            sc_train = scores[train]
            presels  = pre_featsel (X_train, y_train, prefsmethod, prefsthr)
            X_train = X_train[:, presels > 0]
            X_test  = X_test [:, presels > 0]

        #creating grid search pipeline
        if fsname != 'stats' and fsname != 'none':
            #fsp   = append_to_keys(fsp, fsname + '__')
            pipe   = Pipeline([ ('fs', fsmethod), ('cl', classif) ])
            clap   = append_to_keys(clp, 'cl__')
            fisp   = append_to_keys(fsp, 'fs__')
            params = dict(clap.items() + fisp.items())
            gs     = GridSearchCV (pipe, params, n_jobs=n_cpus, verbose=0)
        else:
            gs     = GridSearchCV (classif, clp, n_jobs=n_cpus, verbose=0)

        if fsname == 'univariate':
            gs.fit(X_train, sc_train)
        else:
            gs.fit(X_train, y_train)

        #save predictions
        preds [fc] = gs.predict(X_test)

        train_pred = gs.predict(X_train)

        #AUC score based on training classification
        roc_auc = 0
        if hasattr(classif, 'predict_proba'):
            rprobs = gs.predict_proba(X_train)

            rfpr, rtpr, rthresholds = roc_curve(y_train, rprobs[:, 1], 1)
            roc_auc = auc(rfpr, rtpr)

            probs [fc] = rprobs
        else:
            rfpr, rtpr, rthresholds = roc_curve(y_train, train_pred, 1)
            roc_auc = auc(rfpr, rtpr)

        rscore [fc] = roc_auc
        f1score[fc] = f1_score(y_train, train_pred)
        #save other parameters
        best_p[fc] = gs.best_params_
        truth [fc] = y_test

        fc += 1

    #results[roinom] = classification_metrics (y, preds, probs)
    results['clfmethod']        = clfmethod
    results['cv']               = cv
    results['cvgrid']           = clp
    results['preds']            = preds
    results['truth']            = truth
    results['probs']            = probs
    results['best_params']      = best_p
    results['train_auc_scores'] = rscore
    results['train_f1_scores']  = f1score

    #saving results
    if not outdir:
        outdir = datadir

    n_folds = str(n_folds)
    if cvfold != 'cv':
        n_folds = ''

    outfname = os.path.join(outdir, 'test_' + cvfold + n_folds + '_' + clfmethod + '_' + feats)
    if prefsmethod != 'none':
        outfname += '_' + prefsmethod + str(prefsthr)
    outfname += '_' + fsname

    #np.savez (outfname + '.npz', results)
    #np.save  (outfname + '.npy', results)
    of = open(outfname + '.pickle', 'w')
    pickle.dump (results, of)
    of.close()

    #inf = open(outfname + '.pickle', 'r')
    #res = pickle.load(inf)

#-------------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())

#        gs.fit = <bound method GridSearchCV.fit of GridSearchCV(cv=None,
#       estimator=Pipeline(steps=[('fs', RandomizedPCA(copy=True, iterated_power=3, n_components=None, random_state=0,
#       whiten=False)), ('cl', LinearSVC(C=1.0, class_weight='auto', dual=True, fit_intercept=True,
#     intercept_scaling=1, loss='l2', multi_class='ovr', penalty='l2',
#     random_state=None, tol=0.0001, verbose=0))]),
#       fit_params={}, iid=True, loss_func=None, n_jobs=2,
#       param_grid={'fs__iterated_power': [3, 4, 5], 'cl__C': array([  1.00000e-03,   1.00000e-02,   1.00000e-01,   1.00000e+00,
#         1.00000e+01,   1.00000e+02,   1.00000e+03]), 'fs__whiten': [True, False], 'fs__n_components': [1, 5, 9, 13, 17, 21, 25, 29, 'mle']},


