
import numpy as np
from itertools import permutations

from nipy.modalities.fmri.glm import GeneralLinearModel

from macuto.nifti.read import niftilist_mask_to_array

import logging
log = logging.getLogger(__name__)

class VBMAnalyzer(object):
    """

    """

    def __init__(self):
        """

        :return:
        """
        self.glm_model = None

        self._file_lst = None
        self._labels = None
        self._mask_indices = None
        self._mask_shape = None
        self._x = None
        self._y = None

    @staticmethod
    def _create_group_regressors(labels):
        """

        :param labels: iterable of label values
        label values can be int or string

        :return:
        np.ndarray of zeros and ones with as many columns
        as unique values in labels.
        """
        label_values = np.unique(labels)
        n_subjs = len(labels)
        group_regressors = np.zeros((n_subjs, len(label_values)))
        for lidx, label in enumerate(label_values):
            group_regressors[labels == label, lidx] = 1

        return group_regressors

    def _create_design_matrix(self, labels, regressors=None):
        """
        Returns a VBM group comparison GLM design matrix.
        Concatenating the design matrix corresponding to group comparison
        with given labels and the given regressors, if any.

        :param labels: np.ndarray

        :param regressors: np.ndarray

        :return: np.ndarray

        """
        group_regressors = self._create_group_regressors(labels)

        if regressors is not None:
            group_regressors = np.concatenate((group_regressors, regressors),
                                              axis=1)

        return group_regressors

    @staticmethod
    def _extract_files_from_filedict(file_dict):
        """

        :param file_dict: dict
        file_dict is a dictionary: string/int -> list of file paths

        The key is a string or int representing the group name.
        The values are lists of absolute paths to nifti files which represent
        the subject files (GM or WM tissue volumes)

        :return:

        :todo: give the option to do tissue segmentation
        """
        classes = file_dict.keys()
        if len(classes) < 2:
            msg = 'VBM needs more than one group.'
            log.error(msg)

        file_lst = []
        labels = []
        for idx, group in file_dict:
            file_lst.extend(file_dict[group])
            labels.extend([idx] * len(file_dict[group]))

        return file_lst, labels

    @staticmethod
    def _extract_data_from_volumes(file_lst, mask_file=None):
        """

        :param file_lst:
        :return:
        """
        #create image data matrix
        return niftilist_mask_to_array(file_lst, mask_file)

    def _extract_data(self, file_dict, mask_file=None):
        """

        :param file_dict:
        :param mask:
        :return:
        """
        self._file_lst, \
        self._labels = self._extract_files_from_filedict(file_dict)

        self._y, self._mask_indices, \
        self._mask_shape = self._extract_data_from_volumes(self._file_lst,
                                                           mask_file)

    @staticmethod
    def _nipy_glm(x, y):
        """

        :param x:
        :param y:
        :return:
        """
        myglm = GeneralLinearModel(x)
        myglm.fit(y)
        return myglm

    def _create_group_contrasts(self):
        """

        :return: list arrays
        dict with of contrasts for group comparison
        """
        #create a list of arrays with [1, -1]
        #varying where the -1 is, for each group
        n_groups = len(self._labels)
        contrasts = []
        if n_groups == 2:
            contrasts.append([-1,  1])
            contrasts.append([ 1, -1])

        #if there are 3 groups we have to
        # do permutations of [-1, 0, 1]
        elif n_groups == 3:
            for p in permutations([-1, 0, 1]):
                contrasts.append(p)

        return contrasts

    def fit(self, file_dict, mask_file=None, regressors=None):
        """

        :param filedict:
        :param mask:
        :return:
        """
        #extract masked subjects data matrix from dict of files
        self._extract_data(file_dict, mask_file)

        #create data regressors
        self._x = self._create_design_matrix(self._labels, regressors)

        #fit GeneralLinearModel
        self.glm_model = self._nipy_glm(self._x, self._y)

    def transform(self, contrast_type='t'):
        """
        Apply GLM constrast comparing each group one vs. all.

        :param contrast_type: string
        Defines the type of contrast. See GeneralLinearModel.contrast help.
        choices = {'t', 'F'}

        :return:
        """

        #apply GLM
        # define contrasts
        contrasts = self._create_group_contrasts()

        #http://nbviewer.ipython.org/gist/mwaskom/6263977
        #http://nbviewer.ipython.org/github/jbpoline/bayfmri/blob/master/notebooks/0XX-random-fields.ipynb
        #http://nbviewer.ipython.org/github/jbpoline/bayfmri/blob/master/notebooks/Functional-Connectivity-Nitime.ipynb

        # contrast1 = glm_model.contrast([-1,  1], contrast_type='t')
        # contrast2 = glm_model.contrast([ 1, -1], contrast_type='t')
        #
        # # compute the t-stat
        # ttest1 = contrast1.stat()
        # ttest2 = contrast2.stat()
        #
        # # compute the p-value
        # p1 = contrast1.p_value()
        # p2 = contrast2.p_value()
        # #pvalue005=contrast0.p_value(0.05).shape
        #
        # pvalue005_c1=contrast1.p_value(0.005)
        # pvalue005_c2=contrast2.p_value(0.005)

    def bonferroni_correct(self):


    def grf_correct(self):


    def randomise_correct(self):


    def save_result(self, file_path):
        """
        """
        pass
        #TODO
        # hay que añadir FWE correction como es SPM
        #
        #
        # namep1=os.path.join(outfolder, 'vbm_p0005c1.nii.gz')
        # namep2=os.path.join(outfolder, 'vbm_p0005c2.nii.gz')
        # p005c1=vector_to_volume(pvalue005_c1, mask_indices, mask_shape, dtype=float)
        # p005c2=vector_to_volume(pvalue005_c2, mask_indices, mask_shape, dtype=float)
        # save_niigz(p005c1, namep1, affine=None, header=None)
        # save_niigz(p005c2, namep2, affine=None, header=


    def to_pickle(self):
        pass
        #TODO


    def to_hdf5(self):
        pass
        #TODO


class VBMAnalyzer(VBMAnalyzer):
    """

    """
    #TODO
    from scipy.stats import t

    # http://nbviewer.ipython.org/github/practical-neuroimaging/pna-notebooks/blob/master/GLM_t_F.ipynb
    @staticmethod
    def _t_test(betah, resid, X):
        """
        test the parameters betah one by one - this assumes they are
        estimable (X full rank)

        betah : (p, 1) estimated parameters
        resid : (n, 1) estimated residuals
        X : design matrix
        """

        RSS = sum((resid)**2)
        n = resid.shape[0]
        q = np.linalg.matrix_rank(X)
        df = n-q
        MRSS = RSS/df

        XTX = np.linalg.pinv(X.T.dot(X))

        tval = np.zeros_like(betah)
        pval = np.zeros_like(betah)

        for idx, beta in enumerate(betah):
            c = np.zeros_like(betah)
            c[idx] = 1
            t_num = c.T.dot(betah)
            SE = np.sqrt(MRSS* c.T.dot(XTX).dot(c))
            tval[idx] = t_num / SE

            pval[idx] = 1.0 - t.cdf(tval[idx], df)

        return tval, pval

    @staticmethod
    def _glm(x, y):
        """A GLM function returning the estimated parameters and residuals

        :param X:
        :param Y:
        :return:
        """
        betah   =  np.linalg.pinv(x).dot(y)
        Yfitted =  x.dot(betah)
        resid   =  y - Yfitted
        return betah, Yfitted, resid

    def transform(self):
        """

        :return:
        """
        #TODO
        betah, yfitted, resid = self._glm(self._x, self._y)
        t, p =  self._t_test(betah, resid, self._x)
