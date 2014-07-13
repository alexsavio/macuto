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

from .distance import (DistanceMeasure,
                       PearsonCorrelationDistance,
                       WelchTestDistance,
                       BhatacharyyaGaussianDistance)

from ..threshold import (RobustThreshold,
                         RankThreshold,
                         PercentileThreshold)

from ..utils import Printable
from ..storage import ExportData
from ..exceptions import LoggedValueError


log = logging.getLogger(__name__)


class FeatureSelection(Printable):
    """
    Base class for Feature Selection methods
    """
    pass
#    def select_from(self, X):
#        raise NotImplementedError


class SupervisedFeatureSelection(FeatureSelection):
    """
    Base class for Supervised Feature Selection methods
    """
    pass
#    def select_from(self, X, y):
#        raise NotImplementedError


class DistanceBasedFeatureSelection(SupervisedFeatureSelection):
    """

    """

    def __init__(self, distance_measure, threshold_method):
        """

        :param distance_measure: macuto.classification.distance.DistanceMeasure
        :param threshold_method: macuto.threshold.Threshold
        """
        self._distance_measure = distance_measure
        self._threshold_method = threshold_method

    def fit_transform(self, x, y):
        """

        :param X: np.ndarray
         n_samps x n_feats

        :param y: vector
         n_samps class labels

        :return: thresholded distance measures
        """

        self._distances = self._distance_measure.fit_transform(x, y)
        self._thresholded = self._threshold_method.fit_transform(x)

        return self._thresholded


def feature_selection(X, y, method, thr=95, dist_function=None,
                      thr_method='robust'):
    """
    INPUT
    X             : data ([n_samps x n_feats] matrix)
    y             : class labels
    method        : distance measure: 'pearson', 'bhattacharyya', 'welcht', ''
                    if method == '' or None, will try to use dist_function
    thr           : percentile distance threshold
    dist_function :
    thr_method    : method for thresholding: None, 'robust', 'ranking',
                                             'percentile'

    OUTPUT
    m          : distance measure (thresholded or not)
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
        distance = WelchTestDistance()

    else:
        if dist_function is not None:
            log.info ("Calculating {0} distance between data and "
                      "class labels".format(dist_function.__name__))
            #http://docs.scipy.org/doc/scipy/reference/spatial.distance.html
            distance = DistanceMeasure(dist_function)
        else:
            raise LoggedValueError('Not valid argument input values.')

    dists = distance.fit_transform(X, y)

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

    feats  = np.zeros((n_subjs, 7))

    feats[:, 0] = data.max (axis=1)
    feats[:, 1] = data.min (axis=1)
    feats[:, 2] = data.mean(axis=1)
    feats[:, 3] = data.var (axis=1)
    feats[:, 4] = np.median      (data, axis=1)
    feats[:, 5] = stats.kurtosis (data, axis=1)
    feats[:, 6] = stats.skew     (data, axis=1)

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


def create_feature_sets(fsmethod, data, msk, y, outdir, outbasename,
                        otype='.h5'):
    """
    @param fsmethod:

    @param fsgrid:

    @param data: numpy array
    Shape: n_samples x n_features

    @param msk:

    @param y:

    @param outdir:

    @param outbasename:

    @param otype: string
    Valid values '.pyshelf', '.mat', '.hdf5' or '.h5'

    @return:
    """
    np.savetxt(os.path.join(outdir, outbasename + '_labels.txt'), y,
               fmt="%.2f")

    outfname = os.path.join(outdir, outbasename)
    log.info('Creating ' + outfname)

    fs = data[:, msk > 0]

    if fsmethod == 'stats':
        feats = calculate_stats(fs)

    elif fsmethod == 'hist3d':
        feats = calculate_hist3d(fs)

    elif fsmethod == 'none':
        feats = fs

    #save file
    ExportData.save_variables(outfname + otype, {'feats': feats})
