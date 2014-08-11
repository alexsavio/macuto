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

import numpy as np
from scipy import stats
from collections import OrderedDict
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler

from ..utils.printable import Printable
from .sklearn_utils import (get_pipeline,
                            get_cv_method)

from .results import (ClassificationResult, ClassificationMetrics,
                      classification_metrics, get_cv_classification_metrics,
                      enlist_cv_results_from_dict)

log = logging.getLogger(__name__)


#Classification Pipeline
class ClassificationPipeline(Printable):
    """This class wraps a classification pipeline with grid search.
    Here you can select up to two feature selection methods, one feature
    scaler method, one classifier. It is also possible to specify
    the cross-validation method and the grid search objective function.

    This class uses all functions that are in .sklearn_utils.py and
    .results.py. If you need more details on what choices you can use for
    each pipeline parameter, please have a look there.

    Parameters
    ----------

    clfmethod: str
        See get_classification_algorithm for possible choices

    n_feats: int
        Number of features of the input dataset. This is useful for
        adjusting the feature selection and classification grid search
        parameters.

    fsmethod1: str, optional
        See get_fsmethod for possible choices

    fsmethod2: str, optional
        See get_fsmethod for possible choices

    fsmethod1_kwargs:
        See get_fsmethod for possible choices

    fsmethod2_kwargs:
        See get_fsmethod for possible choices

    scaler: sklearn scaler object

    clfmethod_kwargs:
        See get_classification_algorithm for possible choices

    cvmethod  : string or int
        String with a number or number for K, for a K-fold method.
        'loo' for LeaveOneOut

    stratified: bool
        Indicates whether to use a Stratified K-fold approach

    n_cpus: int
        Number of CPUS to be used in the Grid Search

    gs_scoring: str
        Grid search scoring objective function.
    """

    def __init__(self, clfmethod, n_feats, fsmethod1=None, fsmethod2=None,
                 fsmethod1_kwargs={}, fsmethod2_kwargs={}, clfmethod_kwargs={},
                 scaler=StandardScaler(), cvmethod='10', stratified=True,
                 n_cpus=1, gs_scoring='accuracy'):

        self.n_feats = n_feats
        self.fsmethod1 = fsmethod1
        self.fsmethod2 = fsmethod2
        self.clfmethod = clfmethod

        self.fsmethod1_kwargs = fsmethod1_kwargs
        self.fsmethod2_kwargs = fsmethod2_kwargs
        self.clfmethod_kwargs = clfmethod_kwargs

        self._pipe = None
        self._params = None
        self._cv = None
        self._gs = None
        self._results = None
        self._metrics = None

        self.cvmethod = cvmethod
        self.stratified = stratified
        self.scaler = scaler
        self.n_cpus = n_cpus
        self.gs_scoring = gs_scoring

        self.reset()

    def reset(self):
        """Remakes the pipeline and the gridsearch objects.

        You can use this to modify parameters of this object and this will call
         the necessary functions to remake the pipeline.
        """

        self._pipe = None
        self._params = None
        self._cv = None
        self._gs = None
        self._results = None
        self._metrics = None

        self._pipe, self._params = get_pipeline(self.fsmethod1, self.fsmethod2,
                                                self.clfmethod,
                                                self.n_feats,
                                                self.n_cpus,
                                                self.fsmethod1_kwargs,
                                                self.fsmethod2_kwargs,
                                                self.clfmethod_kwargs)

        #creating grid search
        self._gs = GridSearchCV(self._pipe, self._params, n_jobs=self.n_cpus,
                                verbose=0, scoring=self.gs_scoring)

    def cross_validation(self, samples, targets, cv=None):
        """Performs a cross-validation against a dataset and its labels.

        Parameters
        ----------
        samples: array_like

        targets: vector or list
            Class labels set in the same order as in samples

        Returns
        -------
        Classification_Results, Classification Metrics
        """
        if cv is None:
            self._cv = get_cv_method(targets, self.cvmethod, self.stratified)
        else:
            self._cv = cv

        self.n_feats = samples.shape[1]

        #We use dictionaries to save each fold classification result
        #because we will need to identify all sets of results to one fold.
        #If we used lists, we would loose track of folds if something went
        #wrong.
        preds      = OrderedDict()
        probs      = OrderedDict()
        truth      = OrderedDict()
        best_pars  = OrderedDict()
        importance = OrderedDict()

        fold_count = 0
        for train, test in self._cv:
            log.info('Processing fold ' + str(fold_count))

            #data cv separation
            x_train, x_test, \
            y_train, y_test = samples[train, :], samples[test, :], \
                              targets[train], targets[test]

            # We correct NaN values in x_train and x_test
            nan_mean = stats.nanmean(x_train)
            nan_train = np.isnan(x_train)
            nan_test = np.isnan(x_test)

            #remove Nan values
            x_test[nan_test] = 0
            x_test = x_test + nan_test*nan_mean

            x_train[nan_train] = 0
            x_train = x_train + nan_train*nan_mean

            #y_train = y_train.ravel()
            #y_test = y_test.ravel()

            #scaling
            #if clfmethod == 'linearsvc' or clfmethod == 'onevsonesvc':
            if self.scaler is not None:
                log.info('Normalizing data with: {}'.format(str(self.scaler)))
                x_train = self.scaler.fit_transform(x_train)
                x_test = self.scaler.transform(x_test)

            #do it
            log.info('Running grid search')
            self._gs.fit(x_train, y_train)

            log.info('Predicting on test set')

            #predictions
            preds[fold_count] = self._gs.predict(x_test)
            truth[fold_count] = y_test
            best_pars[fold_count] = self._gs.best_params_

            #features importances
            if hasattr(self._gs.best_estimator_, 'support_vectors_'):
                imp = self._gs.best_estimator_.support_vectors_
            elif hasattr(self._gs.best_estimator_, 'feature_importances_'):
                imp = self._gs.best_estimator_.feature_importances_
            else:
                imp = None

            importance[fold_count] = imp

            #best grid-search parameters
            try:
                probs[fold_count] = self._gs.predict_proba(x_test)
            except Exception as exc:
                probs[fold_count] = None

            log.info('Result: {} classifies as {}.'.format(y_test,
                                                           preds[fold_count]))

            fold_count += 1

        #summarize results
        self._results = ClassificationResult(preds, probs, truth, best_pars,
                                             self._cv, importance, targets)

        #calculate performance metrics
        self._metrics = self.result_metrics()

        return self._results, self._metrics

    def result_metrics(self, classification_results=None):
        """Return the Accuracy, Sensitivity, Specificity, Precision, F1-Score
        and Area-under-ROC of given classification results or self._results
        if None.

        Parameters
        ----------
        classification_results: results.Classification_Result

        Returns
        -------
        If self.cvmethod is 'loo' then return the average values for
        each measure cited above in a results.Classification_Metrics object.

        Otherwhise return two results.Classification_Metrics, the first is the
         average and the second, the standard deviations.
        """
        cr = classification_results

        if cr is None:
            if self._results is None:
                log.error('Cross-validation should be performed before this.')
                return None
            else:
                cr = self._results

        if self.cvmethod == 'loo':
            targets, preds, \
            probs, labels = enlist_cv_results_from_dict(cr.cv_targets,
                                                        cr.predictions,
                                                        cr.probabilities)

            acc, sens, spec, \
            prec, f1, auc = classification_metrics(targets, preds, probs,
                                                   labels)

            return ClassificationMetrics(acc, sens, spec, prec, f1, auc)

        else:
            metrics = get_cv_classification_metrics(cr.cv_targets,
                                                    cr.predictions,
                                                    cr.probabilities)

            avg_metrics = metrics.mean(axis=0)
            std_metrics = metrics.std(axis=0)

            avgs = ClassificationMetrics(**tuple(avg_metrics))
            stds = ClassificationMetrics(**tuple(std_metrics))

            return avgs, stds



