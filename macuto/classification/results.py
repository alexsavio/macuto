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
import collections
import numpy as np

#scores
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


class Result (collections.namedtuple('Result', ['metrics', 'cl', 'prefs_thr',
                                                'subjsf', 'presels', 'prefs',
                                                'fs1', 'fs2', 'y_true', 'y_pred'])):
    pass


def classification_metrics (targets, preds, probs=None, labels=None):
    """
    @param targets:
    @param preds:
    @param probs:
    @param labels:
    @return:
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
        if (cm[0,0] + cm[0,1]) != 0:
            spec = float(cm[0,0])/(cm[0,0] + cm[0,1])

    return acc, sens, spec, prec, f1, auc


def enlist_cv_results(cv_targets, cv_preds, cv_probs=None):
    """
    @param cv_targets:
    @param cv_preds:
    @param cv_probs:
    @return:
    targets, preds, probs, labels
    """
    targets = []
    preds   = []
    probs   = []

    if (isinstance(cv_targets, dict)):
        c = 0
        for i in list(cv_targets.keys()):
            try:
                targets.append(cv_targets[i])
                preds.append  (cv_preds  [i])

                if cv_probs != None:
                    if len(cv_probs) > 0:
                        probs.append(cv_probs  [i])
            except:
                print("Unexpected error: ", sys.exc_info()[0])
            c += 1

    else:
        for i in np.arange(cv_targets.shape[0]):
            targets.append(cv_targets[i,:])
            preds.append  (cv_preds  [i,:])

            if cv_probs != None:
                probs.append(cv_probs[i,:,:])

    if   cv_probs == None  : probs = None
    elif len(cv_probs) == 0: probs = None

    labels = np.unique(targets[0])

    return targets, preds, probs, labels


def get_cv_classification_metrics (cv_targets, cv_preds, cv_probs=None):
    """
    Returns a matrix of size [n_folds x 6],
    where 6 are: acc, sens, spec, prec, f1, roc_auc

    @param cv_targets:
    @param cv_preds:
    @param cv_probs:
    @return:
    """

    targets, preds, probs, labels = enlist_cv_results(cv_targets, cv_preds, cv_probs)

    metrics = np.zeros((len(targets), 6))

    for i in range(len(targets)):
        y_true = targets[i]
        y_pred = preds  [i]

        y_prob = None
        if probs != None:
            y_prob = probs[i]

        acc, sens, spec, prec, f1, roc_auc = classification_metrics (y_true, y_pred, y_prob, labels)
        metrics[i, :] = np.array([acc, sens, spec, prec, f1, roc_auc])

    return metrics


def get_cv_significance(cv_targets, cv_preds):
    """
    Calculates the mean significance across the significance of each
    CV fold confusion matrix.

    Parameters
    ----------
    @param cv_targets:
    @param cv_preds:

    Returns
    -------
    p_value : float
        P-value, the probability of obtaining a distribution at least as extreme
        as the one that was actually observed, assuming that the null hypothesis
        is true.

    Notes
    -----
    I doubt this is a good method of measuring the significance of a classification.

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



def get_confusion_matrix_fisher_significance (table, alternative='two-sided'):
    """
    Returns the value of fisher_exact test on table.

    Parameters
    ----------
    @param table : array_like of ints
        A 2x2 contingency table. Elements should be non-negative integers.

    @param alternative : {'two-sided', 'less', 'greater'}, optional
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
