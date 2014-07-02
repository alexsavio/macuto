import logging

import numpy as np
from scipy import stats
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler

from .features import distance_computation
from ..utils import Printable
from ..threshold import Threshold
from ..exceptions import LoggedError

from .sklearn_utils import (get_pipeline,
                            get_cv_method)


log = logging.getLogger(__name__)



class FeatureSelection(Printable):

    def select_from(self, X, y):
        raise NotImplementedError


class DistanceMeasure(Printable):

    def fit_transform(self):


class DistanceBasedFeatureSelection(FeatureSelection):
    """

    """
    def __init__(self, distance_function, threshold):
        raise NotImplementedError

    def select_from(self, x, y):
        raise NotImplementedError


class ScipyDistanceFeatureSelection(DistanceBasedFeatureSelection):
    """
    This can use any distance function in Scipy:
     http://docs.scipy.org/doc/scipy/reference/spatial.distance.html

    """

    def __init__(self, distance_function, threshold):
        """
        :param dist_function: function
        distance function

        :return:
        """
        if not hasattr(dist_function, '__call__'):
            raise ValueError('dist_function must be a function')

        import scipy.spatial.distance as scipy_dist
        if not hasattr(scipy_dist, dist_function.__name__):
            raise ValueError('dist_function must be a scipy.spatial.distance '
                             'function.')

        self._dist_function = dist_function

    def select_from(self, x, y):
        """
        Apply any given 1-D distance function to x and y.
        Have a look at:
        http://docs.scipy.org/doc/scipy/reference/spatial.distance.html

        :param x: numpy array
        Shape: n_samples x n_features

        :param y: numpy array or list
        Size: n_samples

        :return: numpy array
        Size: n_features
        """
        n_feats = x.shape[1]

        #creating output volume file
        p = np.zeros(n_feats)

        #calculating dist_function across all subjects
        for i in list(range(x.shape[1])):
            p[i] = self._dist_function(x[:, i], y)[0]

        p[np.isnan(p)] = 0

        return p



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

class ClassificationPipeline(Printable):
    """

    """

    def __init__(self, n_feats, fsmethod, clfmethod,
                 prefsmethod=None, prefsthr=None,
                 cvmethod='10', stratified=True, stddize=True,
                 thrmethod='robust', n_cpus=1, gs_scoring='accuracy'):
        """
        """
        Printable.__init__(self)

        self.n_feats = n_feats
        self.prefsmethod = prefsmethod
        self.prefsthr = prefsthr
        self.fsmethod = fsmethod
        self.clfmethod = clfmethod
        self.cvmethod = cvmethod
        self.stratified = stratified
        self.stddize = stddize
        self.thrmethod = thrmethod
        self.n_cpus = n_cpus
        self.gs_scoring = gs_scoring


    def reset(self):
        self._pipe, self._params = get_pipeline(self.fsmethod, 'none',
                                                self.clfmethod, self.n_feats,
                                                self.n_cpus)
        #creating grid search
        self._gs = GridSearchCV(self._pipe, self._params, n_jobs=self.n_cpus,
                                verbose=0, scoring=self.gs_scoring)

        self._prefs = None
        if self._prefsmethod is not None:
            self._prefs = get_prefsmethod(self.prefsthr)


    def cross_validation(self, X, y):

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

    return preds, probs, best_pars, presels, cv, importance, y, truth

    def get_results(self):
