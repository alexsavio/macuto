
import logging
import numpy as np
from itertools import permutations

from nipy.modalities.fmri.glm import GeneralLinearModel

from .nifti.read import vector_to_volume
from .nifti.sets import NiftiSubjectsSet
from .nifti.storage import save_niigz

log = logging.getLogger(__name__)


class VBMAnalyzer(object):
    """

    """

    def __init__(self):
        """

        :return:
        """
        self.glm_model = None

        self._subj_files = None
        self._mask_file = None
        self._labels = []
        self._smooth_mm = None
        self._smooth_mask = False

        self._mask_indices = None
        self._mask_shape = None
        self._x = None
        self._y = None

        self._corrected_pvalues = []

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

    def _create_design_matrix(self, regressors=None):
        """
        Returns a VBM group comparison GLM design matrix.
        Concatenating the design matrix corresponding to group comparison
        with given labels and the given regressors, if any.

        :param labels: np.ndarray

        :param regressors: np.ndarray

        :return: np.ndarray

        """
        group_regressors = self._create_group_regressors(self._labels)

        if regressors is not None:
            try:
                group_regressors = np.concatenate((group_regressors, regressors),
                                                  axis=1)
            except Exception as exc:
                log.exception('Error creating the regressors matrix.')

        return group_regressors

    def _extract_files_from_filedict(self, file_dict):
        """

        :param file_dict: dict
        file_dict is a dictionary: string/int -> list of file paths

        The key is a string or int representing the group name.
        The values are lists of absolute paths to nifti files which represent
        the subject files (GM or WM tissue volumes)

        :param mask_file: str

        :todo: give the option to do tissue segmentation
        """
        classes = file_dict.keys()
        if len(classes) < 2:
            raise ValueError('VBM needs more than one group.')

        self._subj_files = NiftiSubjectsSet(file_dict, self._mask_file)
        self._labels = self._subj_files.labels
        #self._determine_labels()

    # def _determine_labels(self):
    #     """
    #
    #     :return:
    #     """
    #     #self._labels = self._subj_files.labels
    #     # self._label_values = np.unique(self._labels)
    #     #
    #     # self._label_values = {}
    #     # unique = np.unique(self._labels)
    #     # for idx, u in enumerate(unique):
    #     #     self._label_values[u] = idx

    def _extract_data(self, file_dict, mask_file=None, smooth_mm=None,
                      smooth_mask=False):
        """

        :param file_dict:
        :param mask_file:
        :param smooth_mm:
        :param smooth_mask:
        """
        self._smooth_mm = smooth_mm
        self._smooth_mask = smooth_mask
        self._mask_file = mask_file

        self._extract_files_from_filedict(file_dict)

        self._y, self._mask_indices, \
        self._mask_shape = self._subj_files.to_matrix(smooth_mm=self._smooth_mm,
                                                      smooth_mask=self._smooth_mask)

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

    @property
    def n_groups(self):
        return len(self._labels)

    def _create_group_contrasts(self, test_type='t'):
        """

        :param test_type: str
        Choices: 't' or 'F'

        :return: list arrays
        dict with of contrasts for group comparison
        """
        if test_type == 't':
            return self._create_ttest_contrast()
        elif test_type == 'F':
            return self._create_Ftest_contrast()
        else:
            log.error('test_type not understood. Got {0}.'.format(test_type))
            raise NotImplementedError

    def _create_Ftest_contrast(self):
        """

        :return:
        """
        #create a list of arrays with [1, -1]
        #varying where the -1 is, for each group
        if self.n_groups == 2:
            contrasts = [ 1, -1]

        #if there are 3 groups we have to
        # do permutations of [-1, 0, 1]
        elif self.n_groups == 3:
            contrasts = [-1, 0, 1]

        else:
            log.error('Too many groups for contrasts: '
                      '{0}.'.format(self.n_groups))
            raise NotImplementedError

        return contrasts

    def _create_ttest_contrast(self):
        """

        :return:
        """
        #create a list of arrays with [1, -1]
        #varying where the -1 is, for each group
        contrasts = []
        if self.n_groups == 2:
            contrasts.append([ 1, -1])
            contrasts.append([-1,  1])

        #if there are 3 groups we have to
        # do permutations of [-1, 0, 1]
        elif self.n_groups == 3:
            for p in permutations([-1, 0, 1]):
                contrasts.append(p)

        else:
            log.error('Too many groups for contrasts: '
                      '{0}.'.format(self.n_groups))
            raise NotImplementedError

        return contrasts

    def fit(self, file_dict, smooth_mm=4, mask_file=None, regressors=None):
        """

        :param file_dict: dict
        file_dict is a dictionary: string/int -> list of file paths

        :param smooth_mm: int
        gaussian kernel size (smooth_size in mm, not voxels)

        The key is a string or int representing the group name.
        The values are lists of absolute paths to nifti files which represent
        the subject files (GM or WM tissue volumes)

        :param mask_file: str
        Path to a mask file of the same shape as the files in file_dict

        :param regressors: np.array
        Array of size [n_subjs x n_regressors]

        """
        try:
            #extract masked subjects data matrix from dict of files
            self._extract_data(file_dict, mask_file, smooth_mm)

            #create data regressors
            self._x = self._create_design_matrix(regressors)

            #fit GeneralLinearModel
            self.glm_model = self._nipy_glm(self._x, self._y)
        except Exception as exc:
            log.exception('Error creating the data for the GLM.')
            raise

    def transform(self, contrast_type='t', correction_type='fwe'):
        """
        Apply GLM constrast comparing each group one vs. all.

        :param contrast_type: string
        Defines the type of contrast. See GeneralLinearModel.contrast help.
        choices = {'t', 'F'}

        :param correction_type: string
        choices = {'bonferroni', 'fwe', 'fdr'}

        :return:
        """

        #apply GLM
        # define contrasts
        contrasts = self._create_group_contrasts(contrast_type)

        #http://nbviewer.ipython.org/gist/mwaskom/6263977
        #http://nbviewer.ipython.org/github/jbpoline/bayfmri/blob/master/notebooks/0XX-random-fields.ipynb
        #http://nbviewer.ipython.org/github/jbpoline/bayfmri/blob/master/notebooks/Functional-Connectivity-Nitime.ipynb

        self._contrasts = []
        for contrast_vector in contrasts:
            self._contrasts.append(self._nipy_glm.contrast(contrast_vector,
                                                           contrast_type=contrast_type))

        if correction_type == 'bonferroni':
            self.bonferroni_correct()
        elif correction_type == 'fwe':
            self.grf_correct()
        elif correction_type == 'fdr':
            self.randomise_correct()
        else:
            log.error('Could not understand given correction_type: '
                      '{0}.'.format(correction_type))
            raise NotImplementedError


        pval_volumes = [vector_to_volume(corrp, self._mask_indices,
                                         self._mask_shape)
                        for corrp in self._corrected_pvalues]

        return pval_volumes

    def bonferroni_correct(self, threshold=0.05):
        """
        :param threshold
        """
        self._corrected_pvalues = []
        for contraster in self._contrasts:
            self._corrected_pvalues.append(contraster.p_value(threshold))

        #contrast1 = self._nipy_glm.contrast(contrasts[0], contrast_type='t')
        #contrast2 = self._nipy_glm.contrast(contrasts[1], contrast_type='t')
        #
        # # compute the t-stat
        #ttest1 = contrast1.stat()
        #ttest2 = contrast2.stat()
        #
        # # compute the p-value
        # p1 = contrast1.p_value()
        # p2 = contrast2.p_value()
        # #pvalue005=contrast0.p_value(0.05).shape
        #
        # pvalue005_c1=contrast1.p_value(0.005)
        # pvalue005_c2=contrast2.p_value(0.005)

    def grf_correct(self):
        pass
        #TODO

    def randomise_correct(self):
        pass
        #TODO

    def save_result(self, file_path):
        """
        """
        if self._corrected_pvalues is None:
            log.error('The results list is empty. So not saving anything.')

        raise NotImplementedError

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


class VBMAnalyzer2(VBMAnalyzer):
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


if __name__ == '__main__':
    #REFERENCES
    #http://nbviewer.ipython.org/github/jbpoline/bayfmri/blob/master/notebooks/010-Multiple_comparison.ipynb
    #http://nbviewer.ipython.org/github/jbpoline/bayfmri/blob/master/notebooks/006-GLM_t_F.ipynb
    #http://nipy.org/nipy/stable/api/generated/nipy.algorithms.statistics.models.glm.html
    #http://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.linalg.lstsq.html
    import os
    from collections import OrderedDict

    def get_files_for_comparison(dirpath, group_sets):
        """

        :param dirpath:
        :param group_sets:
        :return:
        """
        dirfiles = os.listdir(dirpath)
        file_dict = OrderedDict()
        labels = []
        for idx, gs in enumerate(group_sets):
            group_files = []
            for g in gs:
                file_lst = [os.path.join(dirpath, fname) for fname in dirfiles if g in fname]
                #group_files = get_file_list(dirpath, g)
                group_files.extend(file_lst)
                labels.extend([idx] * len(group_files))

            file_dict[str(gs)] = group_files

        return file_dict, labels


    import socket
    hn = socket.gethostname()
    if hn == 'darya':
        infolder="/home/darya/Documents/santiago/vbm/GM_VBM/data4D"
        GMfolder="/home/darya/Documents/santiago/vbm/GM_VBM/vbm_crl_ea_python/data"#"/home/darya/Documents/santiago/vbm/GM_VBM/data3D"
        maskfolder="/home/darya/Documents/santiago/vbm/GM_VBM/mask"
        outfolder="/home/darya/Documents/santiago/vbm/GM_VBM/vbm_crl_ea_python/vbm_out"
        #WMfolder="/home/darya/Documents/santiago/vbm/WM"
    elif hn == 'buccaneer' or hn == 'finn' or hn == 'corsair':
        root = '/home/alexandre/Dropbox/Data/santiago'
        GMfolder = os.path.join(root, 'data3D')
        maskfolder = os.path.join(os.environ['FSLDIR'], 'data', 'standard')

    #outfolder="" -
    # elif hn == 'finn':
    #     from nilearn import datasets
    #     n_subjects = 50
    #     dataset_files = datasets.fetch_oasis_vbm(n_subjects=n_subjects)
    #     age = dataset_files.ext_vars['age'].astype(float)

    maskfile = os.path.join(maskfolder, 'MNI152_T1_2mm_brain_mask.nii.gz')

    #define group comparisons
    group_comparisons = OrderedDict([('Control vs. AD',     ({'crl'}, {'ea'})),
                                     ('Control vs. MCI',    ({'crl'}, {'dcl'})),
                                     ('Control vs. BD',     ({'crl'}, {'tb'})),
                                     ('Control vs. AD+MCI', ({'crl'}, {'ea', 'dcl'})),
                                     ('BD vs. AD',          ({'tb'},  {'ea'})),
                                     ('BD vs. AD+MCI',      ({'tb'},  {'ea', 'dcl'}))])

    comparison = group_comparisons['BD vs. AD']

    #get list of volume files
    file_dict, labels = get_files_for_comparison(GMfolder, comparison)

    from macuto.vbm import VBMAnalyzer
    vbm = VBMAnalyzer().fit(file_dict, smooth_mm=4, mask_file=maskfile,
                            regressors=None)

    #create image data matrix
    #y, mask_indices, mask_shape = niftilist_mask_to_array(file_lst, maskfile)

    #create data regressors
    #x = vbm.create_design_matrix(labels, regressors=None)
