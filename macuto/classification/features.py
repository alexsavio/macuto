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

import os
import numpy as np
import scipy.stats as stats
import logging

from sklearn.feature_selection.base import SelectorMixin
from sklearn.feature_selection.univariate_selection import (_BaseFilter,
                                                            _clean_nans)

from .distance import (welch_ttest, bhattacharyya_dist,
                       DistanceMeasure,
                       PearsonCorrelationDistance,
                       BhatacharyyaGaussianDistance,
                       WelchTestDistance)

from ..threshold import (RobustThreshold, RankThreshold, PercentileThreshold)
from ..utils.printable import Printable
from ..storage import ExportData


log = logging.getLogger(__name__)


class FeatureSelection(Printable):
    """
    Base class for Feature Selection methods
    """
    pass
#    def select_from(self, X):
#        raise NotImplementedError


class SupervisedSelection(FeatureSelection):
    """
    Base class for Supervised Feature Selection methods
    """
    pass
#    def select_from(self, X, y):
#        raise NotImplementedError


class DistanceBasedSelection(_BaseFilter, SelectorMixin):
    """This is a wrapper class for distance measures base selectors.

    I'm using scikit-learn _BaseFilter as base class in order to be able
    to mix my own filters with other filters in a Pipeline.

    For more info:
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/base.py
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/feature_selection/univariate_selection.py
    """

    def __init__(self, score_func, threshold):
        """
        :param score_func:

        :param threshold: Threshold method
        """
        _BaseFilter.__init__(self, score_func)
        self.threshold = threshold

    def _check_params(self, x, y):
        if not 0 <= self.threshold.value <= 1:
            raise ValueError("threhold should be >=0, <=1; got %r"
                             % self.threshold.value)

    def _get_support_mask(self):
        # Cater for NaNs
        if self.threshold == 1:
            return np.ones(len(self.scores_), dtype=np.bool)
        elif self.threshold == 0:
            return np.zeros(len(self.scores_), dtype=np.bool)

        scores = _clean_nans(self.scores_)

        mask = self.threshold.fit_transform(scores)
        ties = np.where(scores == self.threshold.value)[0]
        if len(ties):
            max_feats = len(scores) * self.threshold.value
            kept_ties = ties[:max_feats - mask.sum()]
            mask[kept_ties] = True
        return mask


class PearsonCorrelationSelection(DistanceBasedSelection):
    """Feature selection method based on Pearson's correlation between the
    groups in X, labeled by y.
    """
    def __init__(self, threshold):
        super(PearsonCorrelationSelection, self).__init__(stats.pearsonr,
                                                          threshold)


class WelchTestSelection(DistanceBasedSelection):
    """Feature selection method based on Welch's t-test between the groups
    in X, labeled by y.
    """
    def __init__(self, threshold):
        super(WelchTestSelection, self).__init__(welch_ttest, threshold)


class BhatacharyyaGaussianSelection(DistanceBasedSelection):
    """Feature selection method based on Univariate Gaussian Bhattacharyya
    distance between the groups in X, labeled by y.
    """

    def __init__(self, threshold):
        super(BhatacharyyaGaussianSelection, self).__init__(bhattacharyya_dist,
                                                            threshold)


def feature_selection(samples, targets, method, thr=95, dist_function=None,
                      thr_method='robust'):
    """
    Parameters
    ----------
    samples:
        data ([n_samps x n_feats] matrix)
    targets:
        class labels

    method: str
        distance measure: 'pearson', 'bhattacharyya', 'welcht', ''
        if method == '' or None, will try to use dist_function

    thr: float
        percentile distance threshold from [0, 100]

    dist_function: distance function
        e.g.: any from
        http://docs.scipy.org/doc/scipy/reference/spatial.distance.html

    thr_method: str
        method for thresholding: None, 'robust', 'ranking', 'percentile'

    Returns
    -------
    m:
        distance measure (thresholded or not)
    """

    #pre feature selection, measuring distances
    #Pearson correlation
    if method == 'pearson':
        log.info('Calculating Pearson correlation')
        distance = PearsonCorrelationDistance()

    #Bhattacharyya distance
    elif method == 'bhattacharyya':
        log.info('Calculating Bhattacharyya distance')
        distance = BhatacharyyaGaussianDistance()

    #Welch's t-test
    elif method == 'welcht':
        log.info("Calculating Welch's t-test")
        distance = WelchTestDistance(thr)

    else:
        if dist_function is not None:
            log.info("Calculating {0} distance between data and "
                     "class labels".format(dist_function.__name__))
            #http://docs.scipy.org/doc/scipy/reference/spatial.distance.html
            distance = DistanceMeasure(dist_function)
        else:
            raise ValueError('Not valid argument input values.')

    dists = distance.fit(samples, targets)

    #if all distance values are 0
    if not dists.any():
        log.info("No differences between groups have been found. "
                 "Are you sure you want to continue?")
        return dists

    #threshold data
    threshold = None
    if thr_method == 'robust':
        threshold = RobustThreshold(thr)
    elif thr_method == 'percentile':
        threshold = PercentileThreshold(thr)
    elif thr_method == 'rank':
        threshold = RankThreshold(thr)

    if threshold is not None:
        dists = threshold.fit_transform(dists)

    return dists


def calculate_stats(data):
    """

    @param data: numpy array
    Shape: n_samples x n_features

    @return:
    """
    n_subjs = data.shape[0]

    feats = np.zeros((n_subjs, 7))

    feats[:, 0] = data.max(axis=1)
    feats[:, 1] = data.min(axis=1)
    feats[:, 2] = data.mean(axis=1)
    feats[:, 3] = data.var(axis=1)
    feats[:, 4] = np.median(data, axis=1)
    feats[:, 5] = stats.kurtosis(data, axis=1)
    feats[:, 6] = stats.skew(data, axis=1)

    return feats


def calculate_hist3d(data, bins):
    """

    @param data: numpy array
    Shape: n_samples x n_features

    @param bins:
    @return:
    """
    n_subjs = data.shape[0]

    feats = np.zeros((n_subjs, bins*bins*bins))

    for s in np.arange(n_subjs):
        h, edges = np.histogramdd(data[s, ], bins=(bins, bins, bins))
        feats[s, :] = h.flatten()

    return feats


def create_feature_sets(fsmethod, samples, mask, targets, outdir, outbasename,
                        otype='.h5'):
    """Calculates and saves a feature set in a file.

    Parameters
    ----------
    fsmethod:

    fsgrid:

    data: array_like
        Shape: n_samples x n_features

    msk: array_like

    targets:

    outdir:

    outbasename:

    otype: string
    Valid values '.pyshelf', '.mat', '.hdf5' or '.h5'
    """
    np.savetxt(os.path.join(outdir, outbasename + '_labels.txt'), targets,
               fmt="%.2f")

    outfname = os.path.join(outdir, outbasename)
    log.info('Creating ' + outfname)

    fs = samples[:, mask > 0]

    if fsmethod == 'stats':
        feats = calculate_stats(fs)

    elif fsmethod == 'hist3d':
        feats = calculate_hist3d(fs)

    elif fsmethod == 'none':
        feats = fs

    #save file
    ExportData.save_variables(outfname + otype, {'feats': feats})
