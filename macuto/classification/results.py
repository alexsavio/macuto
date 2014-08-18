# coding=utf-8
#-------------------------------------------------------------------------------

#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#
#2013, Alexandre Manhaes Savio
#Use this at your own risk!
#-------------------------------------------------------------------------------

import sys
import logging
import collections
import numpy as np

#scores
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

log = logging.getLogger(__name__)


#Classification results namedtuple
classif_results_varnames = ['predictions', 'probabilities', 'cv_targets',
                            'best_parameters', 'cv_folds',
                            'features_importance', 'targets', 'labels']


class ClassificationResult(collections.namedtuple('Classification_Result',
                                                  classif_results_varnames)):
    """
    Namedtuple to store classification results.
    """
    pass


#Classification metrics namedtuple
classif_metrics_varnames = ['accuracy', 'sensitivity', 'specificity',
                            'precision', 'f1_score', 'area_under_curve']


class ClassificationMetrics(collections.namedtuple('Classification_Metrics',
                                                   classif_metrics_varnames)):
    """
    Namedtuple to store classifcation CV results metrics.
    """
    pass


# class Result(collections.namedtuple('Result', ['metrics', 'cl', 'prefs_thr',
#                                               'subjsf', 'presels', 'prefs',
#                                               'fs1', 'fs2',
#                                               'y_true', 'y_pred'])):
#    pass


def classification_metrics(targets, preds, probs=None, labels=None):
    """Calculate Accuracy, Sensitivity, Specificity, Precision, F1-Score
    and Area-under-ROC of given classification results.

    Parameters
    ----------
    targets:
    preds:
    probs:
    labels:

    Returns
    -------
    (acc, sens, spec, prec, f1, auc)
    """

#    if probs != None and len(probs) > 0:
#        fpr, tpr, thresholds = roc_curve(targets, probs[:, 1], 1)
#        roc_auc = roc_auc_score(fpr, tpr)
#    else:
#        fpr, tpr, thresholds = roc_curve(targets, preds, 1)
#        roc_auc = roc_auc_score(targets, preds)

    auc = 0
    if len(targets) > 1:
        auc = roc_auc_score(targets, preds)

    cm = confusion_matrix(targets, preds, labels)

    #accuracy
    acc = accuracy_score(targets, preds)

    #recall? True Positive Rate or Sensitivity or Recall
    sens = recall_score(targets, preds)

    #precision
    prec = precision_score(targets, preds)

    #f1-score
    f1 = f1_score(targets, preds, np.unique(targets), 1)

    tnr  = 0.0
    spec = 0.0
    #True Negative Rate or Specificity (tn / (tn+fp))
    if len(cm) == 2:
        if (cm[0, 0] + cm[0, 1]) != 0:
            spec = float(cm[0, 0])/(cm[0, 0] + cm[0, 1])

    return acc, sens, spec, prec, f1, auc


def enlist_cv_results_from_dict(cv_targets, cv_preds, cv_probs=None):
    """Put cv_targets, cv_preds and cv_probs in lists for performance measures.
    Also returns the set of target labels.

    Parameters
    ----------
    cv_targets: dict
    cv_preds: dict
    cv_probs: dict

    Returns
    -------
    targets, preds, probs, labels
    """
    targets = []
    preds = []
    labels = set()

    if cv_probs is None:
        probs = None
    elif len(cv_probs) == 0:
        probs = None
    else:
        probs = []

    #see if it is a LOO result set
    fold_len = 1
    for fold in cv_targets:
        if len(cv_targets[fold]) != fold_len:
            fold_len = len(cv_targets[fold])
            break

    #read results
    for fold in cv_targets:
        try:
            if fold_len == 1:
                trgt = cv_targets[fold][0]
                pred = cv_preds[fold][0]
            else:
                trgt = cv_targets[fold]
                pred = cv_preds[fold]

            targets.append(trgt)
            preds.append(pred)
        except:
            log.exception('Error accessing classification results.')
            raise

        if cv_probs is not None:
            try:
                if fold_len == 1:
                    prob = cv_probs[fold][0]
                else:
                    prob = cv_probs[fold]

                probs.append(prob)
            except:
                log.exception('Error accessing cv_probs.')
                raise

        for t in np.unique(trgt):
            labels.add(t)

    return targets, preds, probs, list(labels)


def enlist_cv_results(cv_targets, cv_preds, cv_probs=None):
    """Put cv_targets, cv_preds and cv_probs in lists for performance measures.
     Also returns the set of target labels.

    Parameters
    ----------
    cv_targets:
    cv_preds:
    cv_probs:

    Returns
    -------
    targets, preds, probs, labels
    """
    if isinstance(cv_targets, dict):
        return enlist_cv_results_from_dict(cv_targets, cv_preds, cv_probs)

    else:
        targets = []
        preds = []
        probs = []
        labels = set()

        try:
            for fold in np.arange(cv_targets.shape[0]):
                targets.append(cv_targets[fold, :])
                preds.append(cv_preds[fold, :])

                for t in np.unique(cv_targets[fold, :]):
                    labels.add(t)

                if cv_probs is not None:
                    probs.append(cv_probs[fold, :, :])
        except:
            log.exception("Error joining CV results.")
            raise

    if cv_probs is None:
        probs = None
    elif len(cv_probs) == 0:
        probs = None

    return targets, preds, probs, list(labels)


def get_cv_classification_metrics(cv_targets, cv_preds, cv_probs=None):
    """
    Returns a matrix of size [n_folds x 6],
    where 6 are: acc, sens, spec, prec, f1, roc_auc

    Parameters
    ----------
    cv_targets:
    cv_preds:
    cv_probs:

    Returns
    -------
    array_like: metrics
    """

    targets, preds, probs, labels = enlist_cv_results(cv_targets,
                                                      cv_preds,
                                                      cv_probs)

    metrics = np.zeros((len(targets), 6))

    for i in range(len(targets)):
        y_true = targets[i]
        y_pred = preds[i]

        y_prob = None
        if probs is not None:
            y_prob = probs[i]

        acc, sens, spec, prec, \
        f1, roc_auc = classification_metrics(y_true, y_pred, y_prob, labels)

        metrics[i, :] = np.array([acc, sens, spec, prec, f1, roc_auc])

    return metrics


def get_cv_significance(cv_targets, cv_preds):
    """
    Calculates the mean significance across the significance of each
    CV fold confusion matrix.

    Parameters
    ----------
    cv_targets:

    cv_preds:

    Returns
    -------
    p_value : float
        P-value, the probability of obtaining a distribution at least as
        extreme as the one that was actually observed, assuming that the null
        hypothesis is true.

    Notes
    -----
    I doubt this is a good method of measuring the significance of a
    classification.

    See a better test here:
    http://scikit-learn.org/stable/auto_examples/plot_permutation_test_for_classification.html

    """

    targets, preds, probs, labels = enlist_cv_results(cv_targets, cv_preds)

    signfs = []
    for i in range(len(targets)):
        y_true   = targets[i]
        y_pred   = preds  [i]
        conf_mat = confusion_matrix(y_true, y_pred, labels)
        signfs.append(get_confusion_matrix_fisher_significance(conf_mat)[1])

    return np.mean(signfs)


def get_confusion_matrix_fisher_significance(table, alternative='two-sided'):
    """
    Returns the value of fisher_exact test on table.

    Parameters
    ----------
    table : array_like of ints
        A 2x2 contingency table. Elements should be non-negative integers.

    alternative : {'two-sided', 'less', 'greater'}, optional
        Which alternative hypothesis to the null hypothesis the test uses.
        Default is 'two-sided'.

    Returns
    -------
    oddsratio : float
        This is prior odds ratio and not a posterior estimate.

    p_value : float
        P-value, the probability of obtaining a distribution at least as extreme
        as the one that was actually observed, assuming that the null hypothesis
        is true.
    """
    from scipy.stats import fisher_exact
    return fisher_exact(table, alternative)
