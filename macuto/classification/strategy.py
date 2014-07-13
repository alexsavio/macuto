# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
#Authors:
# Alexandre Manhaes Savio <alexsavio@gmail.com>
# Darya Chyzhyk <darya.chyzhyk@gmail.com>
# Borja Ayerdi <ayerdi.borja@gmail.com>
# Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
# Neurita S.L.
#
# BSD 3-Clause License
#
# 2014, Alexandre Manhaes Savio
# Use this at your own risk!
#------------------------------------------------------------------------------


import logging
import collections

import numpy as np
from scipy import stats
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from ..utils import Printable
from ..threshold import Threshold
from ..exceptions import LoggedError

from .sklearn_utils import (get_pipeline,
                            get_cv_method)


log = logging.getLogger(__name__)

#Classification results namedtuple
classif_results_varnames = ['preds', 'probs', 'best_pars', 'presels', 'cv',
                            'importance', 'y', 'truth']

class Classification_Result(collections.namedtuple('Classification_Result',
                                                   classif_results_varnames)):
    """
    Namedtuple to store classification results.
    """
    pass

#Classification metrics namedtuple
classif_metrics_varnames = ['accuracy', 'sensitivity', 'specificity',
                            'precision', 'f1_score', 'area_under_curve']

class Classification_Metrics(collections.namedtuple('Classification_Metrics',
                                                    classif_metrics_varnames)):
    """
    Namedtuple to store classifcation CV results metrics.
    """
    pass


#Classification Pipeline
class ClassificationPipeline(Printable):
    """

    """

    def __init__(self, n_feats, fsmethod1, fsmethod2, clfmethod,
                 fsmethod1_kwargs=None, fsmethod2_kwargs=None, clfmethod_kwargs=None,
                 cvmethod='10', stratified=True, stddize=True,
                 thrmethod='robust', n_cpus=1, gs_scoring='accuracy'):
        """
        """
        Printable.__init__(self)

        self.n_feats = n_feats
        self.fsmethod1 = fsmethod1
        self.fsmethod2 = fsmethod2
        self.clfmethod = clfmethod

        self._fsmethod1_kwargs = fsmethod1_kwargs
        self._fsmethod2_kwargs = fsmethod2_kwargs
        self._clfmethod_kwargs = clfmethod_kwargs

        self.cvmethod = cvmethod
        self.stratified = stratified
        self.stddize = stddize
        self.thrmethod = thrmethod
        self.n_cpus = n_cpus
        self.gs_scoring = gs_scoring

    def reset(self):
        """
        """
        self._pipe, self._params = get_pipeline(self.fsmethod1, self.fsmethod2,
                                                self.clfmethod, self.n_feats,
                                                self.n_cpus)
        #creating grid search
        self._gs = GridSearchCV(self._pipe, self._params, n_jobs=self.n_cpus,
                                verbose=0, scoring=self.gs_scoring)

        self._prefs = None
        if self._prefsmethod is not None:
            self._prefs = get_prefsmethod(self.prefsthr)

    def cross_validation(self, X, y):
        """
        """
        self._cv = get_cv_method(y, self.cvmethod, self.stratified)

        n_feats    = X.shape[1]

        presels    = {}
        preds      = {}
        probs      = {}
        truth      = {}
        best_pars  = {}
        importance = {}
        fc = 0
        for train, test in self._cv:
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
            if self.stddize:
                log.info('Standardizing data')
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test  = scaler.transform    (X_test)

            #PRE feature selection
            if self.prefsmethod != 'none':
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
                                    clfmethod, n_feats, n_cpus)

        #creating grid search
        gs = GridSearchCV(pipe, params, n_jobs=n_cpus, verbose=0,
                          scoring=gs_scoring)

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

    self._results = Classification_Result(preds, probs, best_pars, presels, cv, 
                                          importance, y, truth)

    return self._results

    def get_result_metrics(self):
        """
        """
        if not self._results:
            log.error('Results have not been calculated.')
            return None

        self._results.



