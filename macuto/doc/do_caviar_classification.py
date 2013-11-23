#!/usr/bin/python

import os
import re
import sys
import logging
import argparse
import numpy as np
import nibabel as nib
import pickle

import scipy.stats as stats
import scipy.spatial.distance as sdist

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

global log

#python execution:
'''
import os
import sys

sys.path.append('/home/alexandre/Dropbox/Documents/phd/work/aizkolari')
import aizkolari_utils as au

libdir = '/home/alexandre/Dropbox/Documents/phd/work/oasis_feets'

sys.path.append('/home/alexandre/Dropbox/Documents/phd/work/aizkolari')
import aizkolari_utils as au

sys.path.append('/home/alexandre/Dropbox/Documents/phd/work/caviar')
import do_caviar_classification as cav

hn = au.get_hostname()
if hn == 'azteca':
    wd = '/media/data/oasis_aal'
    dd = '/home/alexandre/Dropbox/Documents/phd/work/caviar'
    outd = '/media/data/oasis_hist3d'
elif hn == 'corsair':
    wd = '/media/alexandre/alextenso/work/oasis_svm'
    dd = '/home/alexandre/Dropbox/Documents/phd/work/caviar'
    outd = '/scratch/oasis_hist3d'
elif hn == 'hpmed':
    wd = '/media/alexandre/iba/data/oasis'
    dd = '/home/alexandre/Dropbox/Documents/phd/work/caviar'
    outd = '/media/alexandre/iba/data/oasis'

verbose   = 2
au.setup_logger(verbose, logfname=None)

maskf = os.path.join(libdir, 'MNI152_T1_1mm_brain_mask_dil.nii.gz')

df   = os.path.join(dd, 'oasis_warpfield_hist3d.npy')
labf = os.path.join(dd, 'oasis_warpfield_hist3d_labels.txt')

y = np.loadtxt(labf).astype(int)
data   = np.load(df)

#CAVIAR
n_folds    = 5
n_learners = 20
lambd      = 0.01

y[y == 0] = -1

preds, perfs = cav.do_caviar (data, y, lambd, n_learners, n_folds)

'''

#-------------------------------------------------------------------------------
def set_parser():

    clfmethods   = ['cart', 'gmm', 'rf', 'svm', 'sgd', 'linsvm', 'percep']
    prefsmethods = ['none', 'pearson', 'bhattacharyya', 'welcht']
    fsmethods    = ['stats', 'rfe', 'rfecv', 'univariate', 'fdr', 'fpr', 'extratrees', 'pca', 'rpca', 'lda'] #svmweights

    parser = argparse.ArgumentParser(description='OASIS AAL classification experiment.')
    parser.add_argument('-d', '--dataf',   dest='dataf',     default='', required=True, help='data npy file with a matrix NxD, where N is n_subjs and D n_feats')
    parser.add_argument('-l', '--labelsf',  dest='labelsf',    default='', required=True,   help='list file with labels for each subject in the same order as in the data file.')
    parser.add_argument('-o', '--outdir',    dest='outdir',      default='', required=False,  help='output data directory path. Will use datadir if not set.')
    parser.add_argument('-f', '--feats',     dest='feats',       default='jacs', choices=ftypes, required=True, help='deformation measure type')
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
def get_aal_info(aal_data, roi_idx):
   return aal_data[aal_data[:,3] == str(roi_idx)].flatten()

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
def classification_metrics (targets, preds, probs=None):

    roc_auc = 0
    if probs:
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

    return accuracy, recall, precision, tnr, roc_auc

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
                    'rf'     : RandomForestClassifier(max_depth=None, min_samples_split=1, random_state=None, compute_importances=True),
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

    if fsmethod == 'stats':
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
def parse_subjects_list (fname, datadir=''):
    labels = []
    subjs  = []

    if datadir:
        datadir += os.path.sep

    try:
        f = open(fname, 'r')
        for s in f:
            line = s.strip().split(',')
            labels.append(np.float(line[0]))
            subjf = line[1].strip()
            if not os.path.isabs(subjf):
                subjs.append (datadir + subjf)
            else:
                subjs.append (subjf)
        f.close()

    except:
        log.error( "Unexpected error: ", sys.exc_info()[0] )
        debug_here()
        sys.exit(-1)

    return [labels, subjs]

#-------------------------------------------------------------------------------
def shelve_vars (ofname, varlist):
   mashelf = shelve.open(ofname, 'n')

   for key in varlist:
      try:
         mashelf[key] = globals()[key]
      except:
         log.error('ERROR shelving: {0}'.format(key))

   mashelf.close()

#-------------------------------------------------------------------------------
def append_to_keys (mydict, preffix):
    return {preffix + str(key) : (transform(value) if isinstance(value, dict) else value) for key, value in mydict.items()}

#-------------------------------------------------------------------------------
def calculate_neigh_graph (X, k):
    dists  = sdist.squareform(sdist.pdist(X, 'euclidean'))

    neighs = np.zeros_like(dists)

    #if not dist_thr:
    #    dist_thr = kd.mean()

    #dists[kd >= dist_thr] = 1
    #dists[kd <  dist_thr] = 0
    for i in np.arange(dists.shape[0]):
        idx = np.argsort(dists[i,:])
        neighs[i,idx[:k+1]] = 1

    #should we zero the diagonal and select k+1 neighbours? #
    #That is what we are doing now
    #here Im supposing that the diagonal will always be 1 for all subjects
    neighs[np.diag_indices_from(neighs)] = 20

    return neighs.astype(int), dists

#-------------------------------------------------------------------------------
def calculate_neigh_graph_with_distthr (X, dist_thr=None):
    dists  = sdist.squareform(sdist.pdist(X, 'euclidean'))
    neighs = dists.copy()

    if not dist_thr:
        dist_thr = dists.mean()

    neighs[dists >= dist_thr] = 1
    neighs[dists <  dist_thr] = 0

    return neighs.astype(int), dists

#-------------------------------------------------------------------------------
def random_classify (X, n_learners, ridx=None, rmin=None, rmax=None):

    n_subjs = X.shape[0]
    n_feats = X.shape[1]

    h = np.zeros((n_subjs, n_learners), dtype=int)

    if ridx == None:
        ridx  = np.random.random_integers (0, n_feats-1, n_learners)
    if rmax == None:
        rmax = X.max(axis=0)
    if rmin == None:
        rmin = X.min(axis=0)

    rvals = (rmax[ridx] - rmin[ridx]) * np.random.random(n_learners) + rmin[ridx]

    for i in np.arange(n_subjs):
        x = X[i,:]

        rh = (x[ridx] >= rvals).astype(int)
        rh[rh == 0] = -1

        h[i, :] = rh

    return h, ridx, rmin, rmax

#-------------------------------------------------------------------------------
def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())

#-------------------------------------------------------------------------------
def cross_sum(g):
    n_elems = g.shape[0]
    cs = np.zeros(n_elems, dtype=g.dtype)

    for k in np.arange(n_elems):
        #mask    = np.ones(n_elems).astype(bool)
        #mask[k] = False
        cs  [k] = np.sum(g[k,:]) + np.sum(g[:,k]) - 2

    return cs

#-------------------------------------------------------------------------------
def calculate_weightmat (X, y, h, lambd, k, n_learners):

    #nearest neighbor graph G
    g, dists = calculate_neigh_graph (X, k)
    #g = symmetrize(g)
    #g [g>1] = 1
    S = cross_sum(g)

    #linear equations
    #H = np.dot(h, h.transpose())
    #building the D matrix (page 4)
    n_subjs = X.shape[0]
    dsiz    = n_subjs * n_learners
    D       = np.zeros((dsiz,dsiz))

    #fat diagonal
    for k in range(n_subjs):
        h_k = np.dot(np.atleast_2d(h[k,:]).T, np.atleast_2d(h[k,:]))
        b_k = h_k + 2*lambd*S[k]*np.eye(n_learners)
        ini = k*n_learners
        fin = (k+1)*n_learners
        D[ini:fin,ini:fin] = b_k

    #non diagonal elements
    for i in range(n_subjs):
        for j in range(n_subjs):
            if i != j:
                k_k  = - 2*lambd*g[i,j]*np.eye(n_learners)
                rini = i*n_learners
                rfin = (i+1)*n_learners
                cini = j*n_learners
                cfin = (j+1)*n_learners
                D[rini:rfin,cini:cfin] = k_k

    #b_t
    ymat = np.reshape(np.tile(y, n_learners), (n_learners, len(y)))
    b = (ymat * h.T).flatten()

    #weight matrix solved
    w = np.linalg.solve(D.transpose()+D, 2*b)

    #w = np.linalg.cholesky()
    w = w.reshape(n_learners, len(y))

    return w

#-------------------------------------------------------------------------------
def calculate_k_nearest_neighbors (XA, XB, k):

    dists = sdist.cdist(XA, XB, 'euclidean')

    neighs = np.zeros_like(dists)

    idx = dists.argsort(axis=0)

    for i in np.arange(dists.shape[1]):
        neighs[idx[:k,i],i] = 1

    return neighs.astype(int), dists

#-------------------------------------------------------------------------------

def calculate_nearest_neighbors (XA, XB, dist_thr=None):

    dists = sdist.cdist(XA, XB, 'euclidean')
    kd    = dists.copy()

    if not dist_thr:
        dist_thr = dists.mean()

    dists[kd >= dist_thr] = 0
    dists[kd <  dist_thr] = 1

    return dists.astype(int), kd

#-------------------------------------------------------------------------------
class bunch:
    __init__ = lambda self, **kw: setattr(self, '__dict__', kw)

#-------------------------------------------------------------------------------
def fit_caviar (X, y, lambd, n_neighs=None, n_learners=20):

    n_subjs = X.shape[0]
    n_feats = X.shape[1]

    if not n_neighs:
        n_neighs = int(np.floor(n_subjs * 0.05))

    #random weaklearner
    h, ridx, rmin, rmax = random_classify (X, n_learners)

    w = calculate_weightmat (X, y, h, lambd, n_neighs, n_learners)

    caviar = bunch()
    caviar.weights    = w
    caviar.X          = X
    caviar.n          = n_subjs
    caviar.n_neighs   = n_neighs
    caviar.ridx       = ridx
    caviar.rmin       = rmin
    caviar.rmax       = rmax
    caviar.n_learners = n_learners

    return caviar

#-------------------------------------------------------------------------------
def predict_caviar(caviar, X, dist):

    n_subjs = X.shape[0]

    neighs, dists = calculate_nearest_neighbors (caviar.X, X, dist)

    beta = 1/(np.sum(dists*neighs)/np.sum(neighs))

    alpha  = np.exp(beta * dists) * neighs
    salpha = np.reshape( np.tile(np.sum(alpha, axis=1), n_subjs), (n_subjs, caviar.n)).transpose()

    alpha  = np.divide(alpha, salpha)
    alpha  = np.nan_to_num(alpha)

    a_sk_num = np.exp(beta * dists)

    h = np.zeros(n_subjs, dtype=int)
    for k in range(n_subjs):
        x = X[k,:]

        a_sk_x_num = a_sk_num[neighs[:,k].astype(bool), k]

        a_sk_x = a_sk_x_num / np.sum(a_sk_x_num)

        h_t = random_classify (np.atleast_2d(x), caviar.n_learners, caviar.ridx, caviar.rmin, caviar.rmax)[0]

        h_t = h_t.flatten()

        w_t = caviar.weights[:,neighs[:,k].astype(bool)]

        h_tta = np.reshape(np.tile(h_t, w_t.shape[1]), (w_t.shape[1], n_learners))

        hh_t = np.sum(a_sk_x * np.sum(w_t * h_tta.T, axis=0))

        pred_t = np.sign(hh_t)

        h[k] = pred_t

    return h

#-------------------------------------------------------------------------------
#CAVIAR
def do_caviar (data, y, lambd=0.01, n_learners=20, n_folds=5):

    ct = StratifiedKFold(y, n_folds)

    preds = []

    fc = 0
    for train, test in ct:
        print '.',

        #train and test sets
        try:
            X_train, X_test, y_train, y_test = data[train,:], data[test,:], y[train], y[test]
        except:
            debug_here()

        #extract validation set
        cv = StratifiedKFold(y_train, n_folds - 1)
        for valt, valv in cv:
            try:
                X_valt, X_valv, y_valt, y_valv = X_train[valt,:], X_train[valv,:], y_train[valt], y_train[valv]
            except:
                debug_here()

        #VALIDATION
        #fit model with validation training set
        caviar_valt = fit_caviar (X_valt, y_valt, lambd, n_neighs=None, n_learners=20)

        #search space for neighbors distance during test
        dists      = sdist.squareform(sdist.pdist(X_valt, 'euclidean'))
        dist_range = np.linspace (np.min(dists), np.max(dists), 250)
        best_acc   = 0.0

        for d in dist_range:
            #predict validation test set
            h_valv = predict_caviar (caviar_valt, X_valv, d)

            accuracy, recall, precision, tnr, roc_auc = classification_metrics (y_valv, h_valv)

            if accuracy > best_acc:
                best_d   = d
                best_acc = perfs_valv['accuracy']

        #TEST
        caviar_train = fit_caviar (X_train, y_train, lambd, n_neighs=None, n_learners=20)

        h_test = predict_caviar (caviar_train, X_test, best_d)

        targets.append(y_test)
        preds.append(h_test)

        fc += 1

        #calculating overall performance
        for p in np.arange(len(preds)):
            accuracy, recall, precision, tnr, roc_auc = classification_metrics (targets[i], preds[i])
            accuracies += accuracy
            recalls    += recall
            precisions += precision
            tnrs       += tnr

        accuracies /= len(preds)
        recalls    /= len(preds)
        precisions /= len(preds)
        tnrs       /= len(preds)

    return preds, accuracies, recalls, precisions, tnrs

#-------------------------------------------------------------------------------
def setup_logger (verbosity=1):
    #define log level
    if verbosity == 0:
     lvl = logging.WARNING
    elif verbosity == 1:
     lvl = logging.INFO
    elif verbosity == 2:
     lvl = logging.DEBUG
    else:
     lvl = logging.WARNING

    log = logging.Logger('caviarpy', lvl)

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
    feats       = args.feats.strip()
    outdir      = args.outdir.strip()
    clfmethod   = args.estimator.strip()
    fsname      = args.fsmethod.strip()
    prefsmethod = args.prefsmethod.strip()
    prefsthr    = args.prefsthr

    cvfold      = args.cvfold.strip()
    n_folds     = args.nfolds
    n_cpus      = args.ncpus
    verbose     = args.verbosity

    setup_logger(verbose, logfname=None)

    scale = True

    #label values
    n_class = 2

    #labels
    y = np.loadtxt(labf).astype(int)
    n_subjs = len(y)

    #data
    data    = np.load(df)
    n_feats = data.shape[1]

    if n_subjs != data.shape[0]:
        print ("Labels and data shapes do not coincide.")
        exit(-1)

    #results
    results = {}
    results['y'] = y

    classif, clp = get_clfmethod (clfmethod, n_feats, n_subjs, n_cpus)

    #feature selection method instance
    #fsmethod, fsp = get_fsmethod (fsname, n_feats, n_subjs, n_cpus)

    #CAVIAR
    n_folds    = 5
    n_learners = 20
    lambd      = 0.01
    k          = int(np.floor(n_subjs * 0.05))
    dist_thr   = np.arange(0,250,2)
    #beta = 1 / (average distance between all samples)

    y[y == 0] = -1

    preds, perfs = do_caviar (data, y, lambd, n_learners, n_folds)

    print(perfs)

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


