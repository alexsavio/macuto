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

import numpy as np
import scipy.stats as stats

from ..utils.validation import check_X_y
from ..utils.printable import Printable


class DistanceMeasure(object):
    """Base adapter class to measure distances between groups in
    labeled datasets

    Parameters
    ----------
    score_func : callable
        Function taking two arrays x and y, and returning one array of
        distance scores
    """

    def __init__(self, score_func):
        self.score_func = score_func
        self.scores_ = None

    def fit(self, samples, targets):
        """

        Parameters
        ----------
        samples: array-like, shape = [n_samples, n_features]
            The training input samples.


        targets: array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        samples, targets = check_X_y(samples, targets, ['csr', 'csc', 'coo'])

        if not callable(self.score_func):
            raise TypeError("The score function should be a callable, %s (%s) "
                            "was passed."
                            % (self.score_func, type(self.score_func)))

        self._check_params(samples, targets)
        self.scores_ = np.asarray(self.score_func(samples, targets))

        return self

    def _check_params(self, samples, targets):
        pass


class PearsonCorrelationDistance(DistanceMeasure):
    """
    The absolute Pearson's correlation between each feature in X and the
    class labels in y.

    Parameters
    ----------
    x: numpy array
        Shape: n_samples x n_features

    y: numpy array or list
        Size: n_samples

    Returns
    -------
    array_like
    Size: n_features
    """

    def __init__(self):
        super(PearsonCorrelationDistance, self).__init__(stats.pearsonr)


class WelchTestDistance(DistanceMeasure):
    """
    Welch's t-test between the groups in X, labeled by y.

    Parameters
    ----------
    x: numpy array
        Shape: n_samples x n_features

    y: numpy array or list
        Size: n_samples

    Returns
    -------
    array_like
    Size: n_features
    """

    def __init__(self, threshold):
        super(WelchTestDistance, self).__init__(self, welch_ttest)


class BhatacharyyaGaussianDistance(DistanceMeasure):
    """
    Univariate Gaussian Bhattacharyya distance
    between the groups in X, labeled by y.

    Parameters
    ----------
    x: numpy array
        Shape: n_samples x n_features

    y: numpy array or list
        Size: n_samples

    Returns
    -------
    array_like
    Size: n_features
    """

    def __init__(self):
        super(BhatacharyyaGaussianDistance, self).__init__(self,
                                                           bhattacharyya_dist)


def pearson_correlation(x, y):
    """
    Calculates for each feature in X the
    pearson correlation with y.

    Parameters
    ----------
    x: numpy array
        Shape: n_samples x n_features

    y: numpy array or list
        Size: n_samples

    Returns
    -------
    array_like
    Size: n_features
    """
    return distance_computation(x, y, stats.pearsonr)


def distance_computation(x, y, dist_function):
    """
    Calculates for each feature in X the
    given dist_function with y.

    Parameters
    ----------
    x: numpy array
        Shape: n_samples x n_features

    y: numpy array or list
        Size: n_samples

    dist_function: function
        distance function

    Returns
    -------
    array_like
    Size: n_features

    Note
    ----
    Apply any given 1-D distance function to X and y.
    Have a look at:
    http://docs.scipy.org/doc/scipy/reference/spatial.distance.html
    """
    #number of features
    n_feats = x.shape[1]

    #creating output volume file
    p = np.zeros(n_feats)

    #calculating dist_function across all subjects
    for i in list(range(x.shape[1])):
        p[i] = dist_function(x[:, i], y)[0]

    p[np.isnan(p)] = 0

    return p


def bhattacharyya_dist(x, y):
    """
    Univariate Gaussian Bhattacharyya distance
    between the groups in X, labeled by y.

    Parameters
    ----------
    x: numpy array
        Shape: n_samples x n_features

    y: numpy array or list
        Size: n_samples

    dist_function: function
        distance function

    Returns
    -------
    array_like
    Size: n_features
    """
    classes = np.unique(y)
    n_class = len(classes)
    n_feats = x.shape[1]

    b = np.zeros(n_feats)
    for i in np.arange(n_class):
        for j in np.arange(i + 1, n_class):
            if j > i:
                xi = x[y == i, :]
                xj = x[y == j, :]

                mi = np.mean (xi, axis=0)
                mj = np.mean (xj, axis=0)

                vi = np.var  (xi, axis=0)
                vj = np.var  (xj, axis=0)

                si = np.std  (xi, axis=0)
                sj = np.std  (xj, axis=0)

                d = 0.25 * (np.square(mi - mj) / (vi + vj)) + \
                    0.5 * (np.log((vi + vj) / (2*si*sj)))
                d[np.isnan(d)] = 0
                d[np.isinf(d)] = 0

                b = np.maximum(b, d)

    return b


def welch_ttest(x, y):
    """
    Welch's t-test between the groups in X, labeled by y.

    Parameters
    ----------
    x: numpy array
        Shape: n_samples x n_features

    y: numpy array or list
        Size: n_samples

    dist_function: function
        distance function

    Returns
    -------
    array_like
    Size: n_features
    """
    classes = np.unique(y)
    n_class = len(classes)
    n_feats = x.shape[1]

    b = np.zeros(n_feats)
    for i in np.arange(n_class):
        for j in np.arange(i+1, n_class):
            if j > i:
                xi = x[y == i, :]
                xj = x[y == j, :]
                yi = y[y == i]
                yj = y[y == j]

                mi = np.mean (xi, axis=0)
                mj = np.mean (xj, axis=0)

                vi = np.var  (xi, axis=0)
                vj = np.var  (xj, axis=0)

                n_subjsi = len(yi)
                n_subjsj = len(yj)

                t = (mi - mj) / np.sqrt((np.square(vi) / n_subjsi) +
                                        (np.square(vj) / n_subjsj))
                t[np.isnan(t)] = 0
                t[np.isinf(t)] = 0

                b = np.maximum(b, t)

    return b

if __name__ == '__main__':
    from sklearn.datasets import make_classification

