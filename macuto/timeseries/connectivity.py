# coding=utf-8
#-------------------------------------------------------------------------------
#License GNU/GPL v3
#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#
#2013, Alexandre Manhaes Savio
#Use this at your own risk!
#-------------------------------------------------------------------------------

from collections import OrderedDict
from exceptions import ArithmeticError
import nitime.fmri.io as fio

from ..nifti.roi import get_roilist_from_atlas, extract_timeseries

from .selection import TimeseriesSelectionFactory
from .similarity_measure import *


def create_connectivity_matrix(func_img, atlas,
                               selection_method='eigen', similarity_measure='coherence',
                               TR=2, **kwargs):
    """
    Extract from preprocessed fMRI a functional connectivity matrix given an atlas
    and procedure parameters.

    @param func_img: nibabel SpatialImage
    Time series MRI volume.

    @param atlas: nibabel SpatialImage
    Volume defining the different ROIs.
    It will be accessed from lower to higher ROI number, that's the order in the
    connectivity matrix.
    It must be in the same space as func_vol.

    @param selection_method: string
    Defines the timeseries selection method to be applied within each ROI.
    Choices: 'mean', 'eigen', 'ilsia', 'cca'
             'filtered', 'mean_and_filtered', 'eigen_and_filtered'
    See .macuto.timeseries.selection more information.

    @param similarity_measure: string
    Defines the similarity measure method to be used between selected timeseries.
    Choices: 'crosscorrelation', 'correlation', 'coherence',
             'mean_coherence', 'mean_correlation'
    See .macuto.timeseries.similarity_measure for more information.

    @param TR: float, optional
    Acquisition TR, if different from the one which can be extracted from the
    functional image file header, or 2 if nothing could be read from the file
    header.

    @param kwargs:  dict with the following keys, some are optional

    @var 'normalize': Whether to normalize the activity in each voxel, defaults to
        None, in which case the original fMRI signal is used. Other options
        are: 'percent': the activity in each voxel is converted to percent
        change, relative to this scan. 'zscore': the activity is converted to a
        zscore relative to the mean and std in this voxel in this scan.

    @var 'filter': dict, optional
        If provided with a dict of the form:
        @var 'lb': float or 0
        Filter lower-bound

        @var 'ub': float or None
        Filter upper-bound

        @var 'method': string
        Filtering method
        Choices: 'fourier','boxcar', 'fir' or 'iir'

        Each voxel's data will be filtered into the frequency range [lb, ub] with
        nitime.analysis.FilterAnalyzer, using the method chosen here (defaults
        to 'fir')

        See .macuto.selection FilteredTimeseries doc strings.

    @var 'n_comps'   : int
    The number of components to be selected from each ROI. Default 1.

    @var 'comps_perc': float from [0, 100]
    The percentage of components to be selected from the ROI.
    If set will be used instead of 'n_comps'

    #TODO
    @var 'shifts': int or dict
        For lagged ts generation.
        If provided with an int b, will use the range [-b, b]
        If provided with a dict of the form:
        @var 'lb': int
        @var 'ub': int
        For each value in range(lb, ub+1) a lagged version of each
        extracted ts will be included in the ts set. Default: {'lb': -3, 'ub': +3}

    @note: See macuto.timeseries.FilteredTimeseries docstrings to get more kwargs related
    to selected timeseries filtering methods.

    @return: ndarray
    Functional connectivity matrix of size N x N, where N is the number of
    ROIs in atlas.

    @raise: ArithmeticError
    If func_vol and atlas do not have the same 3D shape.
    """
    if func_img.shape[0:2] != atlas.shape[0:2]:
        raise ArithmeticError('Functional and atlas volumes do not have the shame shape.')

    func_vol = func_img.get_data()
    atlas_vol = atlas.get_data()

    rois = get_roilist_from_atlas(atlas_vol)

    selected_ts = OrderedDict()
    for r in rois:
        #get all time series within this roi r
        tseries = func_vol[atlas == r, :]

        #remove zeroed time series
        tseries = tseries[tseries.sum(axis=1) != 0, :]

        #filtering
        tseries = fio._tseries_from_nifti_helper(None, tseries, TR,
                                                 kwargs.get('filter',    None),
                                                 kwargs.get('normalize', None),
                                                 None)

        selected_ts[r] = select_timeseries(tseries, selection_method, TR, **kwargs)

    return calculate_similarities(selected_ts, similarity_measure, **kwargs)

    #TODO
    # if kwargs.has_key('fekete-wilf'):
    #
    #     kwargs['sel_filter'] = []
    #     kwargs['sel_filter'].append({'lb': 0.01, 'ub': 0.1 , 'method': 'boxcar'})
    #     kwargs['sel_filter'].append({'lb': 0.03, 'ub': 0.06, 'method': 'boxcar'})
    #
    #     repts = get_rois_timeseries(func_vol, atlas, TR, **kwargs)
    #
    #     debug_here()
    #     '''
    #     wd  = '/home/alexandre/Dropbox/Documents/phd/work/cobre/'
    #     tsf = 'ts_test.pyshelf'
    #
    #     #save
    #     import os
    #     au.shelve_varlist(os.path.join(wd,tsf), ['tseries'], [repts])
    #
    #     #load
    #     import os
    #     import shelve
    #     tsf = os.path.join(wd,tsf)
    #
    #     data    = shelve.open(tsf)
    #     tseries = data['tseries']
    #
    #     lag = 3
    #
    #     lag_corrs  =
    #     part_corrs =
    #     for key, val in tseries.iteritems():
    #         #xcorr(x, y=None, maxlags=None, norm='biased'):
    #         corr01 = xcorr(val.data[0,:], val.data[1,:], maxlags=lag) #feat01
    #         corr12 = xcorr(val.data[1,:], val.data[2,:], maxlags=lag) #feat02
    #         corr20 = xcorr(val.data[2,:], val.data[0,:], maxlags=lag) #feat03
    #
    #     '''
    #
    #     '''
    #     We first extracted the average time series from the 116 automated anatomical
    #     labeling ROIs [28],
    #     The resulting time series were filtered into the
    #     0.01-0.1 Hz [22] and 0.03-0.06 Hz frequency bands [30]. For
    #     each time series array - both the filtered and original time series -
    #     we computed the lagged correlations and partial correlations
    #     ranging from +-3TR and also derived the maximal correlation of
    #     the seven. Negative values were set to zero, as well as
    #     autocorrelations. The correlation matrices were thresholded to
    #     leave a fraction a of the strongest connections using
    #     alpha = [.5, .4, ..., .1] to produce 240 graphs (3x2x8x5; fre-
    #     quency bands, linear/partial correlation, seven lags and their
    #     maximum and five thresholds respectively - see Figure S1). From
    #     each resulting connectivity matrix both weighted and binary
    #      global features were harvested. For local measures, we focused on
    #      a subset of these graphs that has been reported to be
    #      discriminative, zero lagged partial correlations in the 0.01-
    #        0.1 Hz band [22], from which both binary and weighted features
    #        were derived for each ROI.
    #     Docs:
    #     /home/alexandre/Dropbox/Documents/phd/articles/plosone_alexsavio_2013/docs/journal.pone.0062867.s001.pdf
    #     /home/alexandre/Dropbox/Documents/phd/articles/plosone_alexsavio_2013/docs/fekete-combining_classification_fmri_neurodiagnostics-2013.pdf
    #     /home/alexandre/Dropbox/Documents/phd/articles/plosone_alexsavio_2013/docs/sato-fmri_connectivity_granger-2010.pdf
    #     '''

    #     cmat = calculate_fekete_wilf_ts_connmatrix (repts, TR, **kwargs)
    # else:
    #     repts = get_rois_timeseries(func_vol, atlas, TR, **kwargs)
    #     #calculate connectivity_matrices
    #     cmat = calculate_ts_connmatrix (repts, **kwargs)
    #
    # return cmat



def select_timeseries(ts_set, selection_method='eigen', TR=2, **kwargs):
    """
    Selects significant timeseries from the dict of sets of timeseries.
    Each item in ts_set will be transformed to one or fewer timeseries.

    @param ts_set: dict

    @param selection_method: string
    Defines the timeseries selection method to be applied within each ROI.
    Choices: 'mean', 'eigen', 'ilsia', 'cca'
             'filtered', 'mean_and_filtered', 'eigen_and_filtered'
    See .macuto.timeseries.selection more information.

    @param TR: int or float
    Repetition time of the acquisition protocol used for the fMRI from
    where ts_set has been extracted.

    @param kwargs: dict with the following keys, some are optional

        @var 'n_comps'   : int
        The number of components to be selected from the set. Default 1.

        @var 'comps_perc': float from [0, 100]
        The percentage of components to be selected from the set, will
        ignore 'n_comps' if this is set.

        #TODO
        @var 'shifts': int or dict
            For lagged ts generation.
            If provided with an int b, will use the range [-b, b]
            If provided with a dict of the form:
            @var 'lb': int
            @var 'ub': int
            For each value in range(lb, ub+1) a lagged version of each
            extracted ts will be included in the ts set. Default: {'lb': -3, 'ub': +3}

    @note: See macuto.timeseries.FilteredTimeseries docstrings to get more kwargs related
    to selected timeseries filtering methods.

    @return: dict
    Dictionary with the same keys as ts_set, where each item in ts_set is
    a transformed/reduced set of timeseries.
    """
    select_ts = TimeseriesSelectionFactory.create_method(selection_method)

    #arguments for  get_rep_ts
    #kwargs['TR'] = TR

    repts = OrderedDict()
    for r, ts in ts_set.iteritems():
        ts_set[r] = select_ts(ts, TR=TR, **kwargs)

    return repts


def calculate_similarities(tseries, similarity_measure, **kwargs):
    """
    Calculate a matrix of correlations/similarities between all timeseries in tseries.

    @param tseries: dict or list of lists of 1D ndarray (timeseries)
    If it is a list or an OrderedDict, it will do a faster approach calculating
    only half connectivity matrix, then symmetrize it.
    If it is a standard dict it will do a full connectivity calculation.

    @param similarity_measure: string
    Defines the similarity measure method to be used between selected timeseries.
    Choices: 'crosscorrelation', 'correlation', 'coherence',
             'mean_coherence', 'mean_correlation'
    See .macuto.timeseries.similarity_measure for more information.

    param @kwargs: dict with the following keys, some are optional

    @return: ndarray
    A NxN ndarray with the connectivity cross-measures between all timeseries
    in ts_set

    """
    simil_measure = SimilarityMeasureFactory.create_method(similarity_measure)

    n_rois = len(tseries)
    cmat = np.zeros((n_rois, n_rois))

    #this will benefit from the ordering of the time series and
    #calculate only half matrix, then sum its transpose
    if isinstance(tseries, list):

        for tsi1, ts1 in enumerate(tseries):
            for tsi2, ts2 in enumerate(tseries, start=tsi1):
                cmat[tsi1, tsi2] = simil_measure(ts1, ts2, **kwargs)

        cmat = cmat + cmat.T
        cmat[np.diag_indices_from(cmat)] /= 2

    #this will calculate the cmat fully without the "symmetrization"
    elif isinstance(tseries, dict):

        for tsi1, ts1 in enumerate(tseries):
            for tsi2, ts2 in enumerate(tseries):
                cmat[tsi1, tsi2] = simil_measure(ts1, ts2, **kwargs)

    return cmat

#TODO?
# def save_connectivity_matrices(funcs, aal_rois, outfile, TR=None, **kwargs):
#     '''
#     Save in output_basename.pyshelf a list of connectivity matrices extracted from
#     the Parameters.
#
#     Parameters
#     ----------
#     funcs : a string or a list of strings.
#            The full path(s) to the file(s) from which the time-series is (are)
#            extracted.
#
#     aal_rois: a string or a list of strings.
#            The full path(s) to the file(s) from which the ROIs is (are)
#            extracted.
#
#     outfile: a string
#            The full path to the pyshelf file containing the list of connectivity
#             matrices produced here. The '.pyshelf' extension will be added.
#
#             If it is empty, the list will be only returned as object.
#
#     TR: float, optional
#         TR, if different from the one which can be extracted from the nifti
#         file header
#
#     Kwargs:
#     ------
#     normalize: Whether to normalize the activity in each voxel, defaults to
#         None, in which case the original fMRI signal is used. Other options
#         are: 'percent': the activity in each voxel is converted to percent
#         change, relative to this scan. 'zscore': the activity is converted to a
#         zscore relative to the mean and std in this voxel in this scan.
#
#     average: bool, optional whether to average the time-series across the
#            voxels in the ROI (assumed to be the first dimension). In which
#            case, TS.data will be 1-d
#
#     filter: dict, optional
#        If provided with a dict of the form:
#
#        {'lb':float or 0, 'ub':float or None, 'method':'fourier','boxcar' 'fir'
#        or 'iir' }
#
#        each voxel's data will be filtered into the frequency range [lb,ub] with
#        nitime.analysis.FilterAnalyzer, using the method chosen here (defaults
#        to 'fir')
#
#     See TimeseriesSelection_Factory and SimilarityMeasure_Factory methods docstrings.
#
#     tssel_method  : defines the ts selection method. Options: 'mean', 'eigen', 'ilsia', 'cca'
#     simil_measure : defines the similarity measure method.
#         Options: 'crosscorrelation', 'correlation', 'coherence',
#                  'mean_coherence', 'mean_correlation'
#
#
#     For ts selection methods:
#     n_comps   : the number of components to be selected from the set. Default 1.
#     comps_perc: the percentage of components to be selected from the set
#
#     For similarity measure methods:
#
#
#     Returns
#     -------
#
#     The list of connectivity matrices in the same order as the input funcs.
#
#     Note
#     ----
#
#     Normalization occurs before averaging on a voxel-by-voxel basis, followed
#     by the averaging.
#
#     '''
#
#     if isinstance(funcs,str):
#         funcs = [funcs]
#
#     if isinstance(aal_rois,str):
#         aal_rois = [aal_rois]
#
#     matrices = []
#
#     #processing ROIs and time series
#     for i, funcf in enumerate(funcs):
#
#         print funcf
#
#         roisf = aal_rois[i]
#
#         fnc_nii = nib.load(funcf)
#         aal_nii = nib.load(roisf)
#
#         fvol = fnc_nii.get_data()
#         avol = aal_nii.get_data()
#
#         if not TR:
#             TR = get_sampling_interval(fnc_nii)
#
#         try:
#             mat = create_conn_matrix(fvol, avol, TR, **kwargs)
#
#         except ArithmeticError as err:
#             print ('Exception on calculate_connectivity_matrix on ' + funcf + ' and ' + roisf)
#             print ("{0}".format(err))
#         except:
#             print("Unexpected error:", sys.exc_info()[0])
#             raise
#
#         matrices.append([funcf, roisf, mat])
#
#     #save all connectivity_matrices
#     if not 'pyshelf' in outfile:
#         outfile += '.pyshelf'
#
#     au.shelve_varlist(outfile, ['matrices'], [matrices])
#
#     return matrices