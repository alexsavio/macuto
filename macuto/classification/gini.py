
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import LeaveOneOut

from .strategy import ClassificationPipeline


class FeaturesGiniIndex(object):
    """This class wraps a classification method to estimate discrimination
     Gini indices from a set of features using an sklearn.ExtraTreesClassifier
    """

    @staticmethod
    def fit_transform(self, samples, targets):
        """Return the average Gini-index for each sample in a LeaveOneOut
        classification Cross-validation test using ExtraTreesClassifier.

        Returns
        -------
        array_like
        Vector of the size of number of features in each sample.
        """

        n_feats = samples.shape[0]

        pipe = ClassificationPipeline(n_feats, fsmethod1=None, fsmethod2=None,
                                      clfmethod='extratrees', cvmethod='loo')

        results, _ = pipe.cross_validation(samples, targets)

        ginis = np.array(list(results.features_importance.values()))

        return ginis.mean(axis=0)


def get_gini_indices(samples, targets):
    """

    :param samples:
    :param targets:
    :return:
    """
    # Leave One Out
    cv = LeaveOneOut(len(targets))
    feat_imp = np.zeros(samples.shape[1])

    for train, test in cv:

        x_train, x_test, \
        y_train, y_test = samples[train, :], samples[test, :], \
                          targets[train], targets[test]

        # We correct NaN values in x_train and x_test
        nan_mean = stats.nanmean(x_train)
        nan_train = np.isnan(x_train)
        nan_test = np.isnan(x_test)

        x_test[nan_test] = 0
        x_test = x_test + nan_test*nan_mean

        x_train[nan_train] = 0
        x_train = x_train + nan_train*nan_mean

        # Compute mean, std and noise for z-score
        std = np.std(x_train, axis=0)
        med = np.mean(x_train, axis=0)
        noise = [np.random.uniform(-1.e-10, 1.e-10) for p in range(0, x_train.shape[1])]

        # Apply Z-score
        x_train = (x_train-med)/(std+noise)
        #x_test = (x_test-med)/(std+noise)

        # RFE
        # http://scikit-learn.org/stable/modules/generated/
        # sklearn.feature_selection.RFECV.html#sklearn.feature_selection.RFECV

        # Classifier type.
        classifier = ExtraTreesClassifier()
        classifier = classifier.fit(x_train, y_train)

        feat_imp += classifier.feature_importances_

    res = np.around(feat_imp/x_train.shape[0], decimals=4)
    return res


def plot_gini_indices(ginis, var_names, comparison_name, 
                      num_vars_to_plot=20):
    """Plots the Gini Indices of the top num_vars_to_plot 
    variables when discriminating the samples according to targets.
    
    Parameters
    ----------
    ginis : np.ndarray
    Shape 1 x M where M is the number of variables
    
    targets: np.ndarray or list
    Shape 1xN target labels
    
    var_names: list of strings
    Names of the variables for plotting, in the same order as in 
    ginis.
    
    comparison_name: string
    Plot base title
    
    num_vars_to_plot: int
    
    """
    if num_vars_to_plot > len(ginis):
        num_vars_to_plot = len(ginis)

    ginis_sort_idx = np.argsort(ginis)[::-1]

    idx_for_plot = ginis_sort_idx[0:num_vars_to_plot]
    sorted_ginis = ginis[idx_for_plot]
    plot_var_names = np.array(var_names)[idx_for_plot]
    
    fig = plt.figure()#figsize=(6, 4))
    ax = plt.subplot(111)

    #plot bars
    plt.bar(range(num_vars_to_plot), sorted_ginis, color="b",
            align="center",
            alpha=0.5,      # transparency
            width=0.5,)      # smaller bar width


    # set height of the y-axis
    #max_y = max(zip(mean_values, variance)) # returns a tuple
    #plt.ylim([0, (max_y[0] + max_y[1]) * 1.1])
    plt.ylim([0, 0.75])
    plt.xlim([-1, num_vars_to_plot])

    # hiding axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
            labelbottom="on", left="off", right="off", labelleft="on")

    # adding custom horizontal grid lines
    for y in np.linspace(0.2, 0.7, 4):
        plt.axhline(y=y, xmin=0, xmax=4,
                    color="gray", linestyle="--", alpha=0.4)

    # remove axis spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    #ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # set axes labels and title
    plt.title("Gini index {}".format(comparison_name),
              horizontalalignment='center',
              fontsize=14)
    plt.xticks(range(num_vars_to_plot), plot_var_names, rotation=90)

    return fig
