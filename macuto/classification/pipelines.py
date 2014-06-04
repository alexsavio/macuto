# coding=utf-8
#-------------------------------------------------------------------------------

#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#
#2013, Alexandre Manhaes Savio
#Use this at your own risk!
#-------------------------------------------------------------------------------

import logging
 
import numpy as np
import nibabel as nib
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV

from .data_io import load_data
from .sklearn_utils import (get_cv_method,
                            get_pipeline)

from .features import (pearson_correlation,
                       bhattacharyya_dist,
                       welch_ttest,
                       distance_computation)

from ..files.names import get_extension
from ..storage import save_variables_to_shelve
from ..threshold import (rank_threshold,
                         percentile_threshold,
                         robust_range_threshold)

log = logging.getLogger(__name__)


def perform_classification(subjsf, labelsf, outfile, datadir, maskf, clfmethod,
                           fsmethod1, fsmethod2, prefsmethod, prefsthr,
                           cvmethod, thrmethod, stratified, stddize, n_cpus):
    """
    @param subjsf: string
    Path to subjects list file

    @param labelsf: string
    Path to labels list file
    Must be in the same order as subjsf

    @param outfile: string
    Path to output report file

    @param datadir: string
    Path to where the data files are in case subjsf does not have
    absolute paths.

    @param maskf: string
    Path to a mask file

    @param clfmethod: string
    Choice for classification method
    choices:

    @param fsmethod1:
    @param fsmethod2:
    @param prefsmethod:
    @param prefsthr:
    @param cvmethod:
    @param thrmethod:
    @param stratified:
    @param stddize:
    @param n_cpus:
    @return:
    """
    #first list of variables to be saved
    shelfnames = ['subjsf', 'maskf', 'datadir', 'prefsmethod',
                  'prefsthr', 'clfmethod', 'fsmethod1', 'fsmethod2', 'stddize']

    scores = None
    if get_extension(subjsf) != '.npy':
        X, y, scores, imgsiz, mask, indices = load_data(subjsf, datadir,
                                                        maskf, labelsf)
    else:
        X = np.load(subjsf)
        y = np.loadtxt(labelsf)
        mask = nib.load(maskf).get_data()
        #indices = np.where(mask > 0)

    preds, probs, best_pars, presels, cv, \
    importance, y, truth = extract_and_classify(X, y, prefsmethod, prefsthr,
                                    fsmethod1, fsmethod2, clfmethod, cvmethod,
                                    stratified, stddize, thrmethod, n_cpus)

    #save results with pickle
    log.info('Saving results in ' + outfile)

    #extend list of variables to be saved
    shelfnames.extend(['y', 'mask', 'indices', 'cv', 'importance',
                       'preds', 'probs', 'scores', 'best_pars',
                       'truth', 'presels'])

    shelfvars = []
    for v in shelfnames:
        shelfvars.append(eval(v))

    save_variables_to_shelve(outfile + '.pyshelf', shelfnames, shelfvars)

    return cv, truth, preds, probs, presels


def pre_featsel(X, y, method, thr=95, dist_function=None, thr_method='robust'):
    """
    INPUT
    X             : data ([n_samps x n_feats] matrix)
    y             : class labels
    method        : distance measure: 'pearson', 'bhattacharyya', 'welcht', ''
                    if method == '', will use dist_function
    thr           : percentile distance threshold
    dist_function :
    thr_method    : method for thresholding: 'none', 'robust', 'ranking'

    OUTPUT
    m          : distance measure (thresholded or not)
    """

    #pre feature selection, measuring distances
    #Pearson correlation
    if method == 'pearson':
        log.info('Calculating Pearson correlation')
        m = np.abs(pearson_correlation(X, y))

    #Bhattacharyya distance
    elif method == 'bhattacharyya':
        log.info('Calculating Bhattacharyya distance')
        m = bhattacharyya_dist(X, y)

    #Welch's t-test
    elif method == 'welcht':
        log.info("Calculating Welch's t-test")
        m = welch_ttest (X, y)

    elif method == '':
        log.info ("Calculating distance between data and class labels")
        #http://docs.scipy.org/doc/scipy/reference/spatial.distance.html
        m = distance_computation(X, y, dist_function)

    #if all distance values are 0
    if not m.any():
        log.info("No differences between groups have been found. "
                 "Are you sure you want to continue?")
        return m

    #threshold data
    if thr_method != 'none':
        if thr_method == 'robust':
            mt = robust_range_threshold(m, thr)
        elif thr_method == 'percentile':
            mt = percentile_threshold(m, thr)
        elif thr_method == 'rank':
            mt = rank_threshold(m, thr)

        return mt

    return m


def extract_and_classify (X, y, prefsmethod, prefsthr,
                          fsmethod1, fsmethod2, clfmethod, cvmethod,
                          stratified, stddize, thrmethod='robust',
                          n_cpus=1, gs_scoring='accuracy'):
    """
    Parameters
    ----------
    X:

    y:

    prefsmethod:

    prefsthr:

    fsmethod1:

    fsmethod2:

    clfmethod:

    cvmethod:

    stratified: bool

    stddize: bool

    thrmethod: string

    n_cpus: int

    gs_scoring: string, callable or None, optional, default: None
        Valid options are ['accuracy', 'adjusted_rand_score', 'average_precision',
        'f1', 'log_loss', 'mean_squared_error', 'precision', 'r2', 'recall', 'roc_auc']
        See GridSearchCV docs for further details.

    Returns
    -------
    preds, probs, best_pars, presels, cv, importance, y, truth

    """

    #classifiers
    #cgrid    = [10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2]
    #if nclass
    #perfmeas = ['Accuracy', 'Precision', 'Recall', 'F1',
    #            'PRBEP', 'ROCArea', 'AvgPrec']

    #defining parameters for classifiers
    #n_class    = len(np.unique(y))
    n_subjs    = X.shape[0]
    n_feats    = X.shape[1]
    #n_selfeats = np.min(n_feats, int(np.floor(n_subjs*0.06)))

    cv = get_cv_method(y, cvmethod, stratified)

    presels    = {}
    preds      = {}
    probs      = {}
    truth      = {}
    best_pars  = {}
    importance = {}
    fc = 0
    for train, test in cv:
        log.info('Processing fold ' + str(fc))

        #data cv separation
        X_train, X_test, \
        y_train, y_test = X[train, :], X[test, :], y[train], y[test]

        # We correct NaN values in x_train and x_test
        nan_mean = stats.nanmean(X_train)
        nan_train = np.isnan(X_train)
        nan_test = np.isnan(X_test)

        X_test[nan_test] = 0
        X_test = X_test + nan_test*nan_mean

        X_train[nan_train] = 0
        X_train = X_train + nan_train*nan_mean

        #y_train = y_train.ravel()
        #y_test = y_test.ravel()

        #scaling
        #if clfmethod == 'linearsvc' or clfmethod == 'onevsonesvc':
        if stddize:
            log.info('Standardizing data')
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform    (X_test)

        #PRE feature selection
        if prefsmethod != 'none':
            presels[fc] = pre_featsel(X_train, y_train,
                                      prefsmethod, prefsthr, thrmethod)
            if not presels[fc].any():
                log.info('No feature survived the {0} '
                         '({1}: {2}) feature selection.'.format(prefsmethod,
                                                                thrmethod,
                                                                prefsthr))
                continue

            X_train = X_train[:, presels[fc] > 0]
            X_test  = X_test [:, presels[fc] > 0]

        pipe, params = get_pipeline(fsmethod1, fsmethod2,
                                    clfmethod, n_subjs, n_feats, n_cpus)

        #creating grid search
        gs = GridSearchCV(pipe, params, n_jobs=n_cpus, verbose=1,
                          scoring=gs_scoring, verbose=False)

        #do it
        log.info('Running grid search')
        gs.fit(X_train, y_train)

        log.info('Predicting on test set')

        #predictions, feature importances and best parameters
        preds     [fc] = gs.predict(X_test)
        truth     [fc] = y_test
        best_pars [fc] = gs.best_params_

        if hasattr(gs.best_estimator_, 'support_vectors_'):
            importance[fc] = gs.best_estimator_.support_vectors_
        elif hasattr(gs.best_estimator_, 'feature_importances_'):
            importance[fc] = gs.best_estimator_.feature_importances_

        if hasattr(gs.estimator, 'predict_proba'):
            try:
                probs[fc] = gs.predict_proba(X_test)
            except:
                probs[fc] = []

        log.info('Result: {0} classifies as {1}.'.format(y_test, preds[fc]))

        fc += 1

    return preds, probs, best_pars, presels, cv, importance, y, truth
