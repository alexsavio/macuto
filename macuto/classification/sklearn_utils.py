# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
#Authors:
# Alexandre Manhaes Savio <alexsavio@gmail.com>
# Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
# Neurita S.L.
#
# BSD 3-Clause License
#
# 2014, Alexandre Manhaes Savio
# Use this at your own risk!
#------------------------------------------------------------------------------

import numpy as np
import logging

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

#other decompositions
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
from sklearn.lda import LDA
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE

#pipelining
from sklearn.pipeline import Pipeline, FeatureUnion

from .features import (PearsonCorrelationSelection,
                       BhatacharyyaGaussianSelection,
                       WelchTestSelection)

from ..threshold import Threshold
from ..strings import append_to_keys

log = logging.getLogger(__name__)


def get_clfmethod(clfmethod, n_feats, **clf_kwargs):
    """Return a classification method and a classifiers parameter grid-search

    Parameters
    ----------
    clfmethod: str
    See get_classification_algorithm for possible choices

    n_feats: int
        Number of features of the input dataset. This is useful for
        adjusting the feature selection and classification grid search
        parameters

    Returns
    -------
    classifier, param_grid
    """
    classifier = get_classification_algorithm(clfmethod, **clf_kwargs)

    param_grid = get_classifier_parameter_grid(clfmethod, n_feats)

    return classifier, param_grid


def get_classification_algorithm(clfmethod, **clf_kwargs):
    """
    Parameters
    ----------
    clfmethod: str
        clfmethod choices: 'cart', 'rf', 'gmm', 'rbfsvm', 'polysvm', 'linsvm',
                           'sgd', 'percep'

    Returns
    -------
    sklearn.classifier
    """

    #classifiers
    classifiers = {'cart': tree.DecisionTreeClassifier(),

                   'rf': RandomForestClassifier(max_depth=None,
                                                min_samples_split=1,
                                                random_state=None,
                                                **clf_kwargs),

                   'extratrees': ExtraTreesClassifier(compute_importances=True,
                                                      oob_score=True),

                   'gmm': GMM(init_params='wc', n_iter=20, **clf_kwargs),

                   'rbfsvm': SVC(probability=True, max_iter=50000,
                                 class_weight='auto', **clf_kwargs),

                   'polysvm': SVC(probability=True, max_iter=50000,
                                  class_weight='auto', **clf_kwargs),

                   'linsvm': LinearSVC(class_weight='auto', **clf_kwargs),

                   'sgd': SGDClassifier(fit_intercept=True,
                                        class_weight='auto', shuffle=True,
                                        n_iter=np.ceil(10**6 / 416),
                                        loss='modified_huber', **clf_kwargs),

                   'percep': Perceptron(class_weight='auto', **clf_kwargs),
                   }

    return classifiers[clfmethod]


def get_classifier_parameter_grid(clfmethod, n_feats):
    """
    Parameters
    ----------
    clfmethod: str
        clfmethod choices: 'cart', 'rf', 'gmm', 'rbfsvm', 'polysvm', 'linsvm',
                           'sgd', 'percep', 'extratrees'

    n_feats: int
        Number of features in the dataset to adjust feature selection adjust
        grid_search parameters.

    Returns
    -------
    classifiers[clfmethod], clgrid[clfmethod]
    """
    #Classifiers parameter values for grid search
    if n_feats < 10:
        max_feats = list(range(1, n_feats, 2))
    else:
        max_feats = list(range(1, 30, 4))
    max_feats.extend([None, 'auto', 'sqrt', 'log2'])

    clgrid = {'cart': dict(criterion=['gini', 'entropy'],
                           max_depth=[None, 10, 20, 30]),

              'rf': dict(n_estimators=[3, 5, 10, 30, 50, 100],
                         max_features=max_feats),

              'gmm': dict(n_components=[2, 3, 4, 5],
                          covariance_type=['spherical', 'tied',
                                           'diag'],
                          thresh=[True, False]),

              'extratrees': dict(n_estimators=[10, 30, 50],
                                 max_features=max_feats),

              # 'svm'  : dict(kernel = ['rbf', 'linear', 'poly'],
              #               C = np.logspace(-3, 3, num=7, base=10),
              #               gamma = np.logspace(-3, 3, num=7, base=10),
              #               coef0 = np.logspace(-3, 3, num=7, base=10)),
              # 'svm'   : dict(kernel = ['rbf', 'poly'],
              #                C = np.logspace(-3, 3, num=7, base=10),
              #                gamma = np.logspace(-3, 3, num=7, base=10),
              #                coef0=np.logspace(-3, 3, num=7, base=10)),

              'rbfsvm': dict(kernel=['rbf'],
                             C=np.logspace(-3, 3, num=7, base=10),
                             gamma=np.logspace(-3, 3, num=7, base=10)),

              'polysvm': dict(kernel=['poly'],
                              C=np.logspace(-3, 3, num=7, base=10),
                              degree=np.logspace(-3, 3, num=7, base=10)),

              'linsvm': dict(C=np.logspace(-3, 3, num=7, base=10)),

              'sgd': dict(loss=['hinge', 'modified_huber', 'log'],
                          penalty=["l1", "l2", "elasticnet"],
                          alpha=np.logspace(-6, -1, num=6, base=10)),

              'percep': dict(penalty=[None, 'l2', 'l1', 'elasticnet'],
                             alpha=np.logspace(-3, 3, num=7, base=10)),
              }

    return clgrid[clfmethod]


def get_fsmethod(fsmethod, n_feats, n_jobs=1, **kwargs):
    """Creates a feature selectin method and a parameter grid-search.

    Parameters
    ----------
    fsmethod: string
        fsmethod choices: 'rfe', 'rfecv', 'univariate', 'fpr', 'fdr',
                      'extratrees', 'pca', 'rpca', 'lda', 'anova',
                      'pearson', 'bhattacharyya', 'welchtest'

    n_feats: int
        Number of features in the dataset to adjust feature selection
        grid_search parameters.

    n_jobs: int

    kwargs:
        @keyword rfe_ste
        @keyword pca_n_comps
        @keyword feats_to_sel
        @keyword threshold_method: see macuto.threshold.Threshold
        @keyword threshold_value_grid

    Returns
    -------
    fsmethods[fsmethod], fsgrid[fsmethod]
    """
    #calculate RFE and RFECV step
    if n_feats <= 20:
        rfe_step = 1
    else:
        rfe_step = 0.05

    rfe_step = kwargs.get('rfe_step', rfe_step)

    threshold_method = kwargs.get('threshold_method', 'robust')

    threshold_value_grid = kwargs.get('threshold_value_grid',
                                      [0.90, 0.95, 0.99])

    thresholds = [Threshold(threshold_method, thr_value) for thr_value in
                  threshold_value_grid]

    #Feature selection procedures
                 #http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
    fsmethods = {'rfe': RFE(estimator=SVC(kernel="linear"),
                            step=rfe_step, n_features_to_select=2),

                 #http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
                 'rfecv': RFECV(estimator=SVC(kernel="linear"),
                                step=rfe_step, loss_func=roc_auc_score), #cv=3, default; cv=StratifiedKFold(n_subjs, 3)

                 #Univariate Feature selection: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html
                 'univariate': SelectPercentile(f_classif, percentile=5),

                 #http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFpr.html
                 'fpr': SelectFpr(f_classif, alpha=0.05),

                 #http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFdr.html
                 'fdr': SelectFdr(f_classif, alpha=0.05),

                 #http://scikit-learn.org/stable/modules/feature_selection.html
                 'extratrees': ExtraTreesClassifier(n_estimators=50,
                                                    max_features='auto',
                                                    n_jobs=n_jobs,
                                                    random_state=0), #compute_importances=True (default)

                 'pca': PCA(n_components='mle'),
                 'rpca': RandomizedPCA(random_state=0),
                 'lda': LDA(),

                  #http://scikit-learn.org/dev/auto_examples/feature_selection_pipeline.html
                 'anova': SelectKBest(f_regression, k=n_feats),
                 'pearson': PearsonCorrelationSelection(thresholds[0]),
                 'bhattacharyya': BhatacharyyaGaussianSelection(thresholds[0]),
                 'welchtest': WelchTestSelection(thresholds[0]),
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

    fsgrid = {'rfe': dict(estimator_params=[dict(C=0.1), dict(C=1),
                                            dict(C=10)],
                          n_features_to_select=feats_to_sel),
              'rfecv': dict(estimator_params=[dict(C=0.1), dict(C=1),
                                              dict(C=10)]),
              'univariate': dict(percentile=[1, 3, 5, 10]),
              'fpr': dict(alpha=[1, 3, 5, 10]),
              'fdr': dict(alpha=[1, 3, 5, 10]),
              'extratrees': dict(n_estimators=[1, 3, 5, 10, 30, 50],
                                 max_features=max_feats),
              'pca': dict(n_components=n_comps_pca,
                          whiten=[True, False]),
              'rpca': dict(n_components=n_comps,
                           iterated_power = [3, 4, 5],
                           whiten=[True, False]),
              'lda': dict(n_components=n_comps),
              'anova': dict(k=n_comps),

              'pearson': dict(threshold=thresholds),
              'bhattacharyya': dict(threshold=thresholds),
              'welchtest': dict(threshold=thresholds),
              }

    return fsmethods[fsmethod], fsgrid[fsmethod]


def get_cv_method(targets, cvmethod='10', stratified=True):
    """Creates a cross-validation object

    Parameters
    ----------
    targets   : list or vector
        Class labels set in the same order as in X

    cvmethod  : string or int
        String of a number or number for a K-fold method, 'loo' for LeaveOneOut

    stratified: bool
        Indicates whether to use a Stratified K-fold approach

    Returns
    -------
    Returns a class from sklearn.cross_validation
    """
    n = len(targets)

    if cvmethod == 'loo':
        return LeaveOneOut(n)

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

    return StratifiedKFold(targets, int(cvmethod))


def get_pipeline(fsmethod1, fsmethod2, clfmethod, n_feats, n_cpus,
                 fs1_kwargs={}, fs2_kwargs={}, clf_kwargs={}):
    """Returns an instance of a sklearn Pipeline given the parameters

    Parameters
    ----------
    fsmethod1: str
        See get_fsmethod docstring for valid values

    fsmethod2: str
        See get_fsmethod docstring for valid values

    clfmethod: str
        See get_clfmethod docstring for valid values

    n_feats: int
        Number of features

    n_cpus: int

    fs1_kwargs: dict

    fs2_kwargs: dict

    clf_kwargs: dict

    Returns
    -------
    pipe, params
    """

    log.info('Preparing pipeline')

    combined_features = None
    if fsmethod1 is not None or fsmethod2 is not None:
        #feature selection pipeline
        fs1n = fsmethod1
        fs2n = fsmethod2

        #informing user
        info = 'Selecting features: FSMETHOD1: ' + fs1n
        if fs2n is not None:
            info += ', FSMETHOD2: ' + fs2n
        log.info(info)

        #union of feature selection processes
        fs1, fs1p = get_fsmethod(fs1n, n_feats, n_cpus, **fs1_kwargs)
        fs1p = append_to_keys(fs1p, fs1n + '__')
        if fs2n is not None:
            fs2, fs2p = get_fsmethod(fs2n, n_feats, n_cpus, **fs2_kwargs)
            fs2p = append_to_keys(fs2p, fs2n + '__')

            combined_features = FeatureUnion([(fs1n, fs1), (fs2n, fs2)])
            fsp = dict(list(fs1p.items()) + list(fs2p.items()))
        else:
            combined_features = FeatureUnion([(fs1n, fs1)])
            fsp = fs1p

    #classifier instance
    classif, clp = get_clfmethod(clfmethod, n_feats, **clf_kwargs)
    #clp     = append_to_keys(clgrid[clfmethod], clfmethod + '__')

    #if clfmethod == 'gmm':
    #    classif.means_ = np.array([X_train[y_train == i].mean(axis=0)
    #                     for i in xrange(n_class)])

    #creating pipeline
    if combined_features is not None:
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
