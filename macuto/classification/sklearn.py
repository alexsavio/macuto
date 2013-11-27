# coding=utf-8
#-------------------------------------------------------------------------------
#License GNU/GPL v3
#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#
#2013, Alexandre Manhaes Savio
#Use this at your own risk!
#-------------------------------------------------------------------------------

import sys
import numpy as np
import logging as log

#classification
from sklearn import tree
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.mixture import GMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron

#feature selection
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectFdr
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import ExtraTreesClassifier

#cross-validation
from sklearn.cross_validation import KFold
from sklearn.cross_validation import LeaveOneOut
from sklearn.cross_validation import StratifiedKFold

#scores
from sklearn.metrics import roc_auc_score
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import precision_score
#from sklearn.metrics import recall_score
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import f1_score

#other decompositions
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
from sklearn.lda import LDA
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE

#pipelining
from sklearn.pipeline import Pipeline, FeatureUnion

from ..strings import append_to_keys


def get_clfmethod (clfmethod, n_feats):
    """
    @param clfmethod: string
    clfmethod choices: 'cart', 'rf', 'gmm', 'rbfsvm', 'polysvm', 'linsvm', 'sgd', 'percep'

    @param n_feats: int
    Number of features in the dataset to adjust feature selection adjust grid_search parameters.

    @return:
    classifiers[clfmethod], clgrid[clfmethod]
    """

    #classifiers
    classifiers = { 'cart'   : tree.DecisionTreeClassifier(random_state = 0),
                    'rf'     : RandomForestClassifier(max_depth=None, min_samples_split=1,
                                                      random_state=None),
                    'gmm'    : GMM(init_params='wc', n_iter=20, random_state=0),
                    'rbfsvm' : SVC (probability=True, max_iter=50000, class_weight='auto'),
                    'polysvm': SVC (probability=True, max_iter=50000, class_weight='auto'),
                    'linsvm' : LinearSVC (class_weight='auto'),
                    'sgd'    : SGDClassifier (fit_intercept=True, class_weight='auto',
                                              shuffle=True, n_iter = np.ceil(10**6 / 416),
                                              loss='modified_huber'),
                    'percep' : Perceptron (class_weight='auto'),
    }

    #Classifiers parameter values for grid search
    if n_feats < 10:
        max_feats = list(range(1, n_feats, 2))
    else:
        max_feats = list(range(1, 30, 4))
    max_feats.extend([None, 'auto', 'sqrt', 'log2'])

    clgrid =      { 'cart'   : dict(criterion = ['gini', 'entropy'], max_depth = [None, 10, 20, 30]),
                    'rf'     : dict(n_estimators = [3, 5, 10, 30, 50, 100], max_features = max_feats),
                    'gmm'    : dict(n_components = [2,3,4,5], covariance_type=['spherical', 'tied', 'diag'],
                                    thresh = [True, False] ),
                    #'svm'  : dict(kernel = ['rbf', 'linear', 'poly'], C = np.logspace(-3, 3, num=7, base=10), gamma = np.logspace(-3, 3, num=7, base=10), coef0 = np.logspace(-3, 3, num=7, base=10)),
                    #'svm'    : dict(kernel = ['rbf', 'poly'], C = np.logspace(-3, 3, num=7, base=10), gamma = np.logspace(-3, 3, num=7, base=10), coef0=np.logspace(-3, 3, num=7, base=10)),
                    'rbfsvm' : dict(kernel = ['rbf'],  C = np.logspace(-3, 3, num=7, base=10),
                                    gamma  = np.logspace(-3, 3, num=7, base=10)),
                    'polysvm': dict(kernel = ['poly'], C = np.logspace(-3, 3, num=7, base=10),
                                    degree = np.logspace(-3, 3, num=7, base=10)),
                    'linsvm' : dict(C = np.logspace(-3, 3, num=7, base=10)),
                    'sgd'    : dict(loss=['hinge', 'modified_huber', 'log'],
                                    penalty=["l1","l2","elasticnet"],
                                    alpha=np.logspace(-6, -1, num=6, base=10)),
                    'percep' : dict(penalty=[None, 'l2', 'l1', 'elasticnet'],
                                    alpha=np.logspace(-3, 3, num=7, base=10)),
    }

    return classifiers[clfmethod], clgrid[clfmethod]


#-------------------------------------------------------------------------------
def get_fsmethod(fsmethod, n_feats, n_jobs=1):
    """
    @param fsmethod: string
        fsmethod choices: 'rfe', 'rfecv', 'univariate', 'fpr', 'fdr',
                      'extratrees', 'pca', 'rpca', 'lda', 'anova'

    @param n_feats: int
    Number of features in the dataset to adjust feature selection adjust grid_search parameters.

    @param n_jobs: int

    @return:
    fsmethods[fsmethod], fsgrid[fsmethod]
    """

    #Feature selection procedures
                                #http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
    fsmethods = { 'rfe'       : RFE(estimator=SVC(kernel="linear"), step=0.05, n_features_to_select=2),
                                #http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
                  'rfecv'     : RFECV(estimator=SVC(kernel="linear"), step=0.05, loss_func=roc_auc_score), #cv=3, default; cv=StratifiedKFold(n_subjs, 3)
                                #Univariate Feature selection: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html
                  'univariate': SelectPercentile(f_classif, percentile=5),
                                #http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFpr.html
                  'fpr'       : SelectFpr (f_classif, alpha=0.05),
                                #http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFdr.html
                  'fdr'       : SelectFdr (f_classif, alpha=0.05),
                                #http://scikit-learn.org/stable/modules/feature_selection.html
                  'extratrees': ExtraTreesClassifier(n_estimators=50, max_features='auto', n_jobs=n_jobs, random_state=0), #compute_importances=True (default)

                  'pca'       : PCA(n_components='mle'),
                  'rpca'      : RandomizedPCA(random_state=0),
                  'lda'       : LDA(),
                                #http://scikit-learn.org/dev/auto_examples/feature_selection_pipeline.html
                  'anova'     : SelectKBest(f_regression, k=n_feats),
    }

    #feature selection parameter values for grid search
    max_feats = ['auto']
    if n_feats < 10:
        feats_to_sel = list(range(2, n_feats, 2))
        n_comps = list(range(1, n_feats, 2))
    else:
        feats_to_sel = list(range(2, 20, 4))
        n_comps = list(range(1, 30, 4))
    max_feats.extend(feats_to_sel)

    n_comps_pca = list(n_comps)
    n_comps_pca.extend(['mle'])

    fsgrid = { 'rfe'       : dict(estimator_params=[dict(C=0.1), dict(C=1), dict(C=10)],
                                  n_features_to_select=feats_to_sel),
               'rfecv'     : dict(estimator_params=[dict(C=0.1), dict(C=1), dict(C=10)]),
               'univariate': dict(percentile=[1, 3, 5, 10]),
               'fpr'       : dict(alpha=[1, 3, 5, 10]),
               'fdr'       : dict(alpha=[1, 3, 5, 10]),
               'extratrees': dict(n_estimators=[1, 3, 5, 10, 30, 50], max_features=max_feats),
               'pca'       : dict(n_components=n_comps_pca, whiten=[True, False]),
               'rpca'      : dict(n_components=n_comps, iterated_power = [3, 4, 5], whiten=[True, False]),
               'lda'       : dict(n_components=n_comps),
               'anova'     : dict(k=n_comps),
    }

    return fsmethods[fsmethod], fsgrid[fsmethod]


def get_cv_method(targets, cvmethod='10', stratified=True):
    """
    Creates a cross-validation object

    Parameters
    ----------
    @param targets   : list
    Class labels set in the same order as in X

    @param cvmethod  : string or int
    String of a number or number for a K-fold method, 'loo' for LeaveOneOut

    @param stratified: bool
     Indicates whether to use a Stratified K-fold approach

    Returns
    -------
    @return cv: Returns a class from sklearn.cross_validation
    """
    n = len(targets)

    if stratified:
        if isinstance(cvmethod, int):
            return StratifiedKFold(targets, cvmethod)
        elif isinstance(cvmethod, str):
            if cvmethod.isdigit():
                return StratifiedKFold(targets, int(cvmethod))
    else:
        if isinstance(cvmethod, int):
            return KFold(n, cvmethod)

        elif isinstance(cvmethod, str):
            if cvmethod.isdigit():
                return KFold(n, int(cvmethod))

    if cvmethod == 'loo':
        return LeaveOneOut(n)

    return StratifiedKFold(targets, int(cvmethod))


def get_pipeline(fsmethod1, fsmethod2, clfmethod, n_samps, n_feats, n_cpus):
    """
    Returns an instance of a sklearn Pipeline given the parameters

    @param fsmethod1: string
    See get_fsmethod docstring for valid values

    @param fsmethod2: string
    See get_fsmethod docstring for valid values

    @param clfmethod: string
    See get_clfmethod docstring for valid values

    @param n_samps: int
    Number of samples or subjects

    @param n_feats: int
    Number of features

    @param n_cpus: int

    @return:
    pipe, params

    """

    log.info('Preparing pipeline')

    combined_features = None
    if fsmethod1 != 'none' or fsmethod2 != 'none':
        #feature selection pipeline
        fs1n = fsmethod1
        fs2n = fsmethod2

        #informing user
        info = 'Selecting features: FSMETHOD1: ' + fs1n
        if fs2n != 'none':
            info +=', FSMETHOD2: ' + fs2n
        log.info(info)

        #union of feature selection processes
        fs1, fs1p = get_fsmethod (fs1n, n_feats, n_samps, n_cpus)
        fs1p = append_to_keys(fs1p, fs1n + '__')
        if fs2n != 'none':
            fs2, fs2p = get_fsmethod(fs2n, n_feats, n_samps, n_cpus)
            fs2p = append_to_keys(fs2p, fs2n + '__')

            combined_features = FeatureUnion([(fs1n, fs1), (fs2n, fs2)])
            fsp = dict(list(fs1p.items()) + list(fs2p.items()))
        else:
            combined_features = FeatureUnion([(fs1n, fs1)])
            fsp = fs1p

    #classifier instance
    classif, clp = get_clfmethod(clfmethod, n_feats, n_samps)
    #clp     = append_to_keys(clgrid[clfmethod], clfmethod + '__')

    #if clfmethod == 'gmm':
    #    classif.means_ = np.array([X_train[y_train == i].mean(axis=0)
    #                     for i in xrange(n_class)])

    #creating pipeline
    if combined_features:
        pipe = Pipeline([('fs', combined_features), ('cl', classif)])

        #arranging parameters for the whole pipeline
        clp = append_to_keys(clp, 'cl__')
        fsp = append_to_keys(fsp, 'fs__')
        params = dict(list(clp.items()) + list(fsp.items()))
    else:
        #pipe does not work
        #pipe = Pipeline([ ('cl', classif) ])
        #arranging parameters for the whole pipeline
        #clp = append_to_keys(clp, 'cl__')
        pipe = classif
        params = clp


    return pipe, params
