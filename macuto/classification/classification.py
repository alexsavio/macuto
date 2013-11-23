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

import os
import sys
import shelve
import numpy as np
import nibabel as nib
import logging as log

from sklearn.grid_search import ParameterGrid
from sklearn.preprocessing import StandardScaler

from .io import load_data
from ..files import get_extension
from ..io import save_varlist_to_shelve


def perform_classification (subjsf, labelsf, outfile, datadir, maskf, clfmethod, fsmethod1,
                            fsmethod2, prefsmethod, prefsthr, cvmethod, thrmethod, stratified, stddize, n_cpus):
    """
    @param subjsf:
    @param labelsf:
    @param outfile:
    @param datadir:
    @param maskf:
    @param clfmethod:
    @param fsmethod1:
    @param fsmethod2:
    @param prefsmethod:
    @param prefsthr:
    @param cvmethod:
    @param thrmethod:
    @param stratified:
    @param stddize:
    @param n_cpus:
    @return:
    """
    #first list of variables to be saved
    shelfnames = ['subjsf', 'maskf', 'datadir', 'prefsmethod', 'prefsthr', 'clfmethod', 'fsmethod1', 'fsmethod2', 'stddize']

    scores = None
    if get_extension(subjsf) != '.npy':
        X, y, scores, imgsiz, mask, indices = load_data(subjsf, datadir, maskf, labelsf)
    else:
        X = np.load(subjsf)
        y = np.loadtxt(labelsf)
        mask = nib.load(maskf).get_data()
        indices = np.where(mask > 0)

    #for COBRE ONLY, removing last two subjects:
    X = X[:-2,:]
    y = y[:-2]

    preds, probs, best_pars, presels, cv, \
    importance, scores, y, truth = extract_and_classify (X, y, scores, prefsmethod, prefsthr, fsmethod1, fsmethod2,
                                                     clfmethod, cvmethod, stratified, stddize, thrmethod, n_cpus)

    #save results with pickle
    log.info('Saving results in ' + outfile)

    #extend list of variables to be saved
    shelfnames.extend([ 'y', 'mask', 'indices', 'cv', 'importance', 'preds', 'probs', 'scores', 'best_pars', 'truth', 'presels'])

    shelfvars = []
    for v in shelfnames:
        shelfvars.append(eval(v))

    save_varlist_to_shelve(outfile + '.pyshelf', shelfnames, shelfvars)

    return cv, truth, preds, probs, presels

def apply_distance_threshold (distances, thr, method='robust'):
    if   method == 'robust':     return au.robust_range_threshold (distances, thr)
    elif method == 'rank':       return au.rank_threshold         (distances, thr)
    elif method == 'percentile': return au.percentile_threshold   (distances, thr)

#-------------------------------------------------------------------------------
def pre_featsel (X, y, method, thr=95, dist_function=None, thr_method='robust'):
    '''
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
    '''

    #pre feature selection, measuring distances
    #Pearson correlation
    if method == 'pearson':
        au.log.info ('Calculating Pearson correlation')
        m = np.abs(pearson_correlation (X, y))

    #Bhattacharyya distance
    elif method == 'bhattacharyya':
        au.log.info ('Calculating Bhattacharyya distance')
        m = bhattacharyya_dist (X, y)

    #Welch's t-test
    elif method == 'welcht':
        au.log.info ("Calculating Welch's t-test")
        m = welch_ttest (X, y)

    elif method == '':
        au.log.info ("Calculating distance between data and class labels")
        #http://docs.scipy.org/doc/scipy/reference/spatial.distance.html
        m = distance_computation(X, y, dist_function)

    #if all distance values are 0
    if not m.any():
        au.log.info("No differences between groups have been found. Are you sure you want to continue?")
        return m

    #threshold data
    if thr_method != 'none':
        if thr_method == 'robust':
            mt = au.robust_range_threshold (m, thr)
        elif thr_method == 'percentile':
            mt = au.percentile_threshold (m, thr)
        elif thr_method == 'rank':
            mt = au.rank_threshold (m, thr)

        return mt

    return m



def extract_and_classify (X, y, scores, prefsmethod, prefsthr, fsmethod1, fsmethod2,
                          clfmethod, cvmethod, stratified, stddize,
                          thrmethod='robust', n_cpus=1, gs_scoring='accuracy'):
    '''
    Parameters
    ----------
    X:

    y:

    scores:

    prefsmethod:

    prefsthr:

    fsmethod1:

    fsmethod2:

    clfmethod:

    cvmethod:

    stratified: bool

    stddize: bool

    thrmethod: string

    n_cpus: int

    gs_scoring: string, callable or None, optional, default: None
        Valid options are ['accuracy', 'adjusted_rand_score', 'average_precision',
        'f1', 'log_loss', 'mean_squared_error', 'precision', 'r2', 'recall', 'roc_auc']
        See GridSearchCV docs for further details.

    Returns
    -------
    preds, probs, best_pars, presels, cv, importance, scores, y, truth

    '''

    #classifiers
    #cgrid    = [10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2]
    #if nclass
    #perfmeas = ['Accuracy', 'Precision', 'Recall', 'F1', 'PRBEP', 'ROCArea', 'AvgPrec']

    #defining parameters for classifiers
    n_class    = len(np.unique(y))
    n_subjs    = X.shape[0]
    n_feats    = X.shape[1]
    n_selfeats = np.min(n_feats, int(np.floor(n_subjs*0.06)))

    cv = get_cv_method(y, cvmethod, stratified)

    presels    = {}
    preds      = {}
    probs      = {}
    truth      = {}
    best_pars  = {}
    importance = {}
    fc = 0
    for train, test in cv:
        au.log.info('Processing fold ' + str(fc))

        #data cv separation
        X_train, X_test, y_train, y_test = X[train,:], X[test,:], y[train], y[test]

        #scaling
        #if clfmethod == 'linearsvc' or clfmethod == 'onevsonesvc':
        if stddize:
            log.info('Standardizing data')
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform    (X_test)
            #[X_train, dmin, dmax] = au.rescale (X_train, scale_min, scale_max)
            #[X_test,  emin, emax] = au.rescale (X_test,  scale_min, scale_max, dmin, dmax)


        #PRE feature selection
        if prefsmethod != 'none':
            #sc_train = scores[train]
            presels[fc] = pre_featsel (X_train, y_train, prefsmethod, prefsthr, thrmethod)
            if not presels[fc].any():
                log.info('No feature survived the ' + prefsmethod + '(' + thrmethod + ': '+ str(prefsthr) + ')' + ' feature selection.')
                continue

            X_train = X_train[:, presels[fc] > 0]
            X_test  = X_test [:, presels[fc] > 0]

        pipe, params = get_pipeline(fsmethod1, fsmethod2, clfmethod, n_subjs, n_feats, n_cpus)

        #creating grid search
        gs = GridSearchCV(pipe, params, n_jobs=n_cpus, verbose=1, scoring=gs_scoring)

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
                probs [fc] = gs.predict_proba(X_test)
            except:
                probs [fc] = []

        #hello user
        au.log.info( 'Result: ' + str(y_test) + ' classified as ' + str(preds[fc]))

        fc += 1

    return preds, probs, best_pars, presels, cv, importance, scores, y, truth


def do_experiment_paramgrid (wd, masks, dataf, labelsf, param_grid, thrmethod, stratified, stddize, cvfolds, n_cpus):

    results = {}
    for j in list(ParameterGrid(param_grid)):

        subjsf    = j['subjsf']
        cl        = j['cl']
        prefs     = j['prefs']
        prefs_thr = j['prefs_thr']
        fs1       = j['fs1']
        fs2       = j['fs2']

        maskf = masks[dataf.index(subjsf)]

        of = os.path.join(wd, au.remove_ext(subjsf))
        if prefs != 'none': of += '_' + prefs + '_' + str(prefs_thr)
        if fs1   != 'none': of += '_' + fs1
        of += '_' + cl

        if   cl == 'rf': stddize = False
        else           : stddize = True

        print("Running " + prefs + " " + str(prefs_thr) + " " + fs1 + "_" + str(j))

        if not os.path.exists(of + '.pyshelf'):
            cv, truth, preds, probs, presels = do_classification (subjsf, labelsf, of, wd, maskf, cl, fs1, fs2, prefs, prefs_thr, cvfolds, thrmethod, stratified, stddize, n_cpus)
        else:
            print ('Previously done.')
            print ('Loading ' + of)
            res = shelve.open(of + '.pyshelf')
            cv, truth, preds, probs, presels = res['cv'], res['truth'], res['preds'], res['probs'], res['presels']

        #save results
        #metrics[c, :] = np.array([acc, sens, spec, prec, f1, roc_auc])
        metrics = aizc.get_cv_classification_metrics (truth, preds, probs)

        j['y_pred']  = preds
        j['y_true']  = truth

        j['metrics'] = metrics
        j['presels'] = presels

        if not results.has_key(subjsf):
            results[subjsf] = []

        results[subjsf].append(Result(**j))

    return results

##==============================================================================
##execution script
'''
import sys
sys.path.append('/home/alexandre/Dropbox/Documents/work/cobre')

import do_cobre_classification as cb
results = cb.do_experiment(2)

cb.plot_all_results (results)

'''
def do_experiment(n_cpus=1):

    au.setup_logger(2)

    temporal_filtering     = [True, False]
    global_nuis_correction = [True, False]
    preprocess_grid = {'temporal_filtering': [True, False], 'global_nuis_correction': [True, False]}

    results = {}
    for j in list(ParameterGrid(preprocess_grid)):
        wd, xd, dd, labelsf, phenof, dataf, masks, dilmasks, templates, pipe = get_filepaths(**j)

        stratified, stddize, cvfolds, thrmethod, clf_methods, prefs_methods, prefs_thrs = get_experiment_parameters()
        param_grid    = {'subjsf': dataf, 'cl': clf_methods, 'prefs' : prefs_methods, 'prefs_thr': prefs_thrs, 'fs1' : ['none'], 'fs2' : ['none']}

        #loop will load all classification results files (or classify again) and calculate performance metrics
        results[str(j)] = do_experiment_paramgrid(wd, masks, dataf, labelsf, param_grid, thrmethod, stratified, stddize, cvfolds, n_cpus)

        #plot_results (results[str(j)], wd, masks, dataf, param_grid, stratified, stddize, cvfolds)

    return results



##==============================================================================
##execution script
'''
import sys
sys.path.append('/home/alexandre/Dropbox/Documents/work/cobre')

import do_cobre_classification as cb
results = cb.do_experiment(2)

sum_tabs = cb.print_all_summary_tables (results)

'''
def print_all_summary_tables(results):
    from do_cobre_featsets import get_filepaths

    temporal_filtering     = [True, False]
    global_nuis_correction = [True, False]
    preprocess_grid = {'temporal_filtering': [True, False], 'global_nuis_correction': [True, False]}

    stratified, stddize, cvfolds, thrmethod, clf_methods, prefs_methods, prefs_thrs = get_experiment_parameters()

    sum_tab = {}
    for j in list(ParameterGrid(preprocess_grid)):

        wd, xd, dd, labelsf, phenof, subjsf, masks, dilmasks, templates, pipe = get_filepaths(**j)

        print str(j)
        print 'Doing pipeline ' + pipe + '\n'

        sum_tab[str(j)] = print_summary_table(results[str(j)], wd, masks, subjsf, prefs_methods, prefs_thrs, clf_methods)

        #prefs_method

    return sum_tab


##==============================================================================
def print_summary_table (results, wd, masks, subjsf, prefs_methods, prefs_thrs, clf_methods):

    ##SHOW FINAL RESULTS
    datatype = [
    'ALFF',
    'fALFF',
    'ReHo',
    'VMHC',
    'VMHC stat',
    ]

    sum_tab = {}
    for fi, f in enumerate(subjsf):

        f = subjsf[fi]
        #print f
        resf = results[f]
        #print len(resf)
        d = datatype[fi]
        sum_tab[d] = {}

        for p in clf_methods:
            #print p
            resfp = aizc.filter_objlist (resf, 'cl', p)

            #print (p + '_' + os.path.basename(f))

            sum_tab[d][p] = {}
            crow = ''
            i = 0
            for c in prefs_methods:
                sum_tab[d][p][c] = {}

                resfpc = aizc.filter_objlist (resfp, 'prefs', c)

                i += 1

                # getting accuracies
                signfs = []
                means, varis = [], []
                m_idx = 0 #metric index
                for t in prefs_thrs:
                    resfpct = aizc.filter_objlist (resfpc, 'prefs_thr', t)[0]

                    #metrics[i, :] = np.array([acc, sens, spec, prec, f1, roc_auc])
                    metrs = np.array(resfpct.metrics)

                    means.append(metrs.mean(axis=0)[m_idx])
                    varis.append(metrs.var (axis=0)[m_idx])

                    signfs.append(aizc.get_cv_significance(resfpct.y_true, resfpct.y_pred))

                max_means = np.max(means)
                max_varis = np.max(varis)

                row   = "%.2f (%.2f)" % (max_means, max_varis)
                crow += '\t' + row
                #row = "%.2f (%.2f)" % (max_means[m_idx], max_varis[m_idx])
                #print (d + ' ' + p + ' ' + c + ' ' + row)
                sum_tab[d][p][c] = row

            print (d + ' ' + p + ' \t (' + str(prefs_methods) + '): ' + crow)

    return sum_tab

##==============================================================================
def get_localizations (X, y, cv, maskf, presels, sv):

    mask, hdr, aff = au.get_nii_data(maskf)
    maskidx = np.array(np.where(mask > 0))
    hdr.set_data_dtype(np.dtype(np.float))

    my_presels  = np.zeros_like(presels[0])
    my_svs      = np.zeros_like(mask)
    my_svs_done = False

    #unmasking method found in:
    #http://nisl.github.io/auto_examples/plot_ica_resting_state.html
    from nisl.io import NiftiMasker

    k = 0
    cv.n = X.shape[0]
    for train,test in cv:

        X_train, y_train = X[train,:], y[train]

        preselsvol = np.zeros_like (mask, dtype=np.float)
        preselsvol[tuple(maskidx)] = presels[k] > 0
        preselsnii = au.save_nibabel ('', preselsvol, aff, hdr)

        my_presels += presels[k] > 0

        if len(sv) > 0:
            try:
                nifti_masker = NiftiMasker(mask=preselsnii)
                nifti_masker.fit (X_train[:,presels[k]>0], y_train)
                niimg = nifti_masker.inverse_transform(sv[k][0])
                #X_masked = nifti_masker.fit_transform (X_train[:,presels[k]>0], y_train) #,y_train, target_affine=aff, target_shape=hdr.get_data_shape()
                #niimg = nifti_masker.inverse_transform(sv[0][0])
                #act = np.ma.masked_array(niimg.get_data(), niimg.get_data() == 0)

                my_svs += niimg.get_data()
                my_svs_done = True
            except:
                pass

        k += 1

    my_presels /= cv.n_folds
    my_svs     /= cv.n_folds

    prelocs = np.zeros_like (mask, dtype=np.float)
    prelocs[tuple(maskidx)] = my_presels

    return prelocs, my_svs, my_svs_done

##==============================================================================
##Calculating and saving localization masks
'''
import sys
sys.path.append('/home/alexandre/Dropbox/Documents/work/cobre')
import do_cobre_classification as cb
from sklearn.grid_search import ParameterGrid

volaxis=2
save_fig=True
overwrite=True

temporal_filtering     = [True, False]
global_nuis_correction = [True, False]
preprocess_grid = {'temporal_filtering': [True, False], 'global_nuis_correction': [True, False]}

for j in list(ParameterGrid(preprocess_grid)):
    cb.do_localization_images(j['temporal_filtering'], j['global_nuis_correction'], 
                        save_fig=save_fig, volaxis=volaxis, overwrite=overwrite)
'''
def do_localization_images (temporal_filtering=True, global_nuis_correction=True, 
                           save_fig=True, volaxis=0, overwrite=False):

    if save_fig:
        import matplotlib
        matplotlib.use('AGG')

    import pylab as plt
    from sklearn.grid_search import ParameterGrid

    sys.path.append('/home/alexandre/Dropbox/Documents/work/visualize_volume')
    import visualize_volume as vis

    #save_fig = True
    #temporal_filtering     = True
    #global_nuis_correction = False
    wd, xd, dd, labelsf, phenof, dataf, masks, dilmasks, templates, pipe = get_filepaths(temporal_filtering, global_nuis_correction)

    au.setup_logger(2, logfname=None)

    stratified, stddize, cvfolds, thrmethod, clf_methods, prefs_methods, prefs_thrs = get_experiment_parameters()
    param_grid    = {'subjsf': dataf, 'cl': clf_methods, 'prefs' : prefs_methods, 'prefs_thr': prefs_thrs, 'fs1' : ['none']}

    #loop will load all classification results dataf (or classify again) and calculate performance metrics
    results = {}
    for j in list(ParameterGrid(param_grid)):

        subjsf    = j['subjsf']
        cl        = j['cl']
        prefs     = j['prefs']
        prefs_thr = j['prefs_thr']
        #fs1 = j['fs1']
        #fs2 = j['fs2']

        of = os.path.join(wd, au.remove_ext(subjsf))
        if prefs != 'none': of += '_' + prefs + '_' + str(prefs_thr)
        #if fs1   != 'none': of += '_' + fs1
        #if fs2   != 'none': of += '_' + fs2
        #of += '_' + cl

        #checking if output files exist
        fig1name = of + '_presels_axis' + str(volaxis) + '.png' #presels are the same for all classifiers

        of = of + '_' + cl

        fig2name = of + '_sup_vecs_axis' + str(volaxis) + '.png'

        #overwrite?
        if not overwrite:
            if os.path.exists(fig1name) and os.path.exists(fig2name):
                print ("File " + fig1name + " already exists, jumping to next.")
                continue

        print("Running " + prefs + " " + str(prefs_thr) + " " + str(j))

        #loading data file
        data_file = of + '.pyshelf'
        print ('Loading ' + data_file)
        res = shelve.open(data_file)

        try:
            presels = res['presels']
            cv      = res['cv']
            sv      = res['importance']
        finally:
            res.close()

        #Loading data within cv in order to unmask feature_importances_
        X = np.load(subjsf)
        y = np.loadtxt(labelsf)

        fidx   = dataf.index(subjsf)
        maskf  = masks[fidx]
        tmpltf = templates[fidx]

        tmplt  = nib.load(tmpltf).get_data()

        #getting localizations from data
        presels_vol, supvecs_vol, supvecs_done = get_localizations (X, y, cv, maskf, presels, sv)

        #preselections figure
        fig1 = vis.show_many_slices(tmplt, presels_vol, volaxis=volaxis, vol2_colormap=plt.cm.hot, figtitle=of + '_presels')
        if save_fig:
            aizc.save_fig_to_png (fig1, fig1name, facecolor='white')
            #call_innercrop (xd, fig1name)
        else:
            raw_input("Press Enter to continue...")

        #feature_importances figure
        if supvecs_done:
            fig2 = vis.show_many_slices(tmplt, supvecs_vol, volaxis=volaxis, vol2_colormap=plt.cm.hot, figtitle=of + '_sup_vecs')
            if save_fig:
                aizc.save_fig_to_png (fig2, fig2name, facecolor='white')
                #call_innercrop(xd, fig2name)
            else:
                raw_input("Press Enter to continue...")

##==============================================================================
def call_innercrop (xd, figname):
    if os.access (os.path.join(xd,'innercrop'), os.X_OK):
        comm = os.path.join(xd,'innercrop') + ' -o white ' + figname + ' ' + figname
        print ('Calling: ' + comm)
        au.sys_call (comm)
    else:
        print('Could not find innercrop for execution.')


##==============================================================================
def call_autoaq (xd, maskf, thresh, atlas, outfile):
    if os.access (os.path.join(xd,'autoaq'), os.X_OK):
        comm = os.path.join(xd,'autoaq') + ' -i ' + maskf + ' -t ' + str(thresh) + ' -a ' + '"' + atlas + '"' + ' -o ' + outfile
        print ('Calling: ' + comm)
        au.exec_command (comm)
    else:
        print('Could not find autoaq for execution.')

##==============================================================================
def call_atlasquery(xd, maskf, atlas, outfile):
    if os.access ('/usr/bin/atlasquery', os.X_OK):
        comm = 'atlasquery' + ' -m ' + maskf + ' -a ' + '"' + atlas + '"' + ' > ' + outfile
        print ('Calling: ' + comm)
        au.exec_command (comm)
    else:
        print('Could not find atlasquery for execution.')

##==============================================================================
def call_atlasquerpy(xd, maskf, atlas, outfile):
    import subprocess
    import numpy as np
    atlasquerpy = '/home/alexandre/Dropbox/Documents/work/atlasquerpy/atlasquerpy'
    if os.access (atlasquerpy, os.X_OK):
        comm = atlasquerpy + ' -t roiover -m ' + maskf + ' -a ' + '"' + atlas + '"' + ' > ' + outfile
        print ('Calling: ' + comm)
        process = subprocess.Popen(comm, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        output, error = process.communicate()
        #subprocess.call (comm, shell=True)
        #run_cmd (comm)
        np.savetxt(outfile, np.array(output.split('\n')[:-1]), fmt='%s')
    else:
        print('Could not find atlasquery for execution.')

##==============================================================================
def get_resultfile_name (subjsf, cl, prefs, prefs_thr, fs1='none', fs2='none'):

    resultf = au.remove_ext(subjsf)
    if prefs != 'none': resultf += '_' + prefs + '_' + str(prefs_thr)
    if fs1   != 'none': resultf += '_' + fs1
    if fs2   != 'none': resultf += '_' + fs2
    resultf += '_' + cl
    resultf += '.pyshelf'

    return resultf

##==============================================================================
def get_data_from_shelf (resultfile_path):
    #loading data file
    res = shelve.open(resultfile_path)

    try:
        presels = res['presels']
        cv      = res['cv']
        sv      = res['importance']
    finally:
        res.close()

    return presels, cv, sv

##==============================================================================
def get_localizations_from_datashelf(resultfile_path, subjsf, labelsf, maskf):

    presels, cv, sv = get_data_from_shelf (resultfile_path)

    #Loading data within cv in order to unmask feature_importances_
    X = np.load(subjsf)
    y = np.loadtxt(labelsf)

    #getting localizations from data
    presels_vol, supvecs_vol, supvecs_done = get_localizations (X, y, cv, maskf, presels, sv)

    return presels_vol

##==============================================================================
##Calculating and saving localization masks
'''
import sys
sys.path.append('/home/alexandre/Dropbox/Documents/work/cobre')
import do_cobre_classification as cb

save_fig=True
overwrite=True

cb.do_tanimoto_comparison(save_fig=save_fig, overwrite=overwrite)
'''
def do_tanimoto_comparison (save_fig=True, overwrite=False):

    if save_fig:
        import matplotlib
        matplotlib.use('AGG')

    import pylab as plt
    from sklearn.grid_search import ParameterGrid

    sys.path.append('/home/alexandre/Dropbox/Documents/work/visualize_volume')
    import visualize_volume as vis

    temporal_filtering     = [True, False]
    global_nuis_correction = [True, False]
    preprocess_grid = {'temporal_filtering': [True, False], 'global_nuis_correction': [True, False]}

    fig = plt.figure('Feature Vectors Tanimoto Distance')
    c = 0

    for j in list(ParameterGrid(preprocess_grid)):

        temporal_filtering, global_nuis_correction = j['temporal_filtering'], j['global_nuis_correction']

        #save_fig = True
        #temporal_filtering     = True
        #global_nuis_correction = False
        wd, xd, dd, labelsf, phenof, dataf, masks, dilmasks, templates, pipe = get_filepaths(temporal_filtering, global_nuis_correction)

        au.setup_logger(2, logfname=None)

        stratified, stddize, cvfolds, thrmethod, clf_methods, prefs_methods, prefs_thrs = get_experiment_parameters()
        param_grid    = {'subjsf': dataf[0:3], 'cl': [clf_methods[0]]}

        prefs_thrs = [95]

        for prefs_m in prefs_methods:

            for prefs_thr in prefs_thrs:

                tani_mat, ticks = tanimoto_distances_matrix (param_grid, temporal_filtering, global_nuis_correction, prefs_m, prefs_thr)

                fname = 'tanimoto_distance_matrix_' + prefs_m + '_' + str(prefs_thr) + '.png'

                ax = plt.subplot(3,4,c)

                ax.imshow(tani_mat, cmap=plt.cm.jet, interpolation='nearest')

                axtit = ''
                if temporal_filtering and global_nuis_correction:
                    axtit += 'TPF-GSR'
                elif temporal_filtering and not global_nuis_correction:
                    axtit += 'TPF'
                elif not temporal_filtering and global_nuis_correction:
                    axtit += 'GSR'

                # Move left and bottom spines outward by 10 points
                ax.spines['left'].set_position(('outward', 10))
                ax.spines['bottom'].set_position(('outward', 10))
                # Hide the right and top spines
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                # Only show ticks on the left and bottom spines
                ax.yaxis.set_ticks_position('left')
                ax.xaxis.set_ticks_position('bottom')

                ax.set_xticks(range(len(ticks)))
                ax.set_yticks(range(len(ticks)))

                ax.set_xticklabels(ticks)
                ax.set_yticklabels(ticks)

                ax.set_title(axtit)

                if c % 4 == 0:
                    ax.set_axis_off()

                c += 1

    fig.show()

##==============================================================================
def tanimoto_distances_matrix (param_grid, temporal_filtering, global_nuis_correction, prefs_method, prefs_thr):

    from scipy.spatial.distance import rogerstanimoto

    wd, xd, dd, labelsf, phenof, dataf, masks, dilmasks, templates, pipe = get_filepaths(temporal_filtering, global_nuis_correction)

    #loop will load all classification results dataf (or classify again) and calculate performance metrics
    param_gridlist = list(ParameterGrid(param_grid))
    n = len(param_gridlist)

    tanis = np.zeros((n,n))
    ticks = []

    for j1idx, j1 in enumerate(param_gridlist):

        subjsf    = j1['subjsf']
        cl        = j1['cl']

        fidx   = dataf.index(subjsf)
        maskf  = masks[fidx]

        result_file1 = os.path.join(wd, get_resultfile_name (subjsf, cl, prefs_method, prefs_thr))
        presels_vol1 = get_localizations_from_datashelf(result_file1, subjsf, labelsf, maskf)

        ticks.append(subjsf.split('_')[1])

        print('Tanimoto comparing ' + result_file1)

        for j2idx, j2 in enumerate(param_gridlist):

            subjsf    = j2['subjsf']
            cl        = j2['cl']

            fidx   = dataf.index(subjsf)
            maskf  = masks[fidx]

            result_file2 = os.path.join(wd, get_resultfile_name (subjsf, cl, prefs_method, prefs_thr))
            presels_vol2 = get_localizations_from_datashelf(result_file2, subjsf, labelsf, maskf)

            print('with ' + result_file2)

            tanis[j1idx, j2idx] = rogerstanimoto(presels_vol1.flatten().astype(bool), presels_vol2.flatten().astype(bool))

    return tanis, ticks

#http://matplotlib.org/examples/ticks_and_spines/spines_demo_dropped.html
##==============================================================================
'''
import sys
sys.path.append('/home/alexandre/Dropbox/Documents/work/cobre')
import do_cobre_classification as cs
from sklearn.grid_search import ParameterGrid

overwrite=True

aq_thresh = 0.99

#atlasquery atlases
atlases = []
atlases.append('MNI Structural Atlas')
atlases.append('Harvard-Oxford Cortical Structural Atlas')
atlases.append('Harvard-Oxford Subcortical Structural Atlas')

temporal_filtering     = [True, False]
global_nuis_correction = [True, False]
preprocess_grid = {'temporal_filtering': [True, False], 'global_nuis_correction': [True, False], 'atlas': atlases}

for j in list(ParameterGrid(preprocess_grid)):
    cs.do_autoaq(j['temporal_filtering'], j['global_nuis_correction'], overwrite, j['atlas'], aq_thresh)
'''
def do_autoaq (temporal_filtering=True, global_nuis_correction=True, overwrite=False, atlas='MNI Structural Atlas', aq_thresh=0.95):

    from sklearn.grid_search import ParameterGrid

    wd, xd, dd, labelsf, phenof, dataf, masks, dilmasks, templates, pipe = get_filepaths(temporal_filtering, global_nuis_correction)

    au.setup_logger(2, logfname=None)

    stratified, stddize, cvfolds, thrmethod, clf_methods, prefs_methods, prefs_thrs = get_experiment_parameters()
    #param_grid    = {'subjsf': dataf, 'cl': clf_methods, 'prefs' : prefs_methods, 'prefs_thr': prefs_thrs, 'fs1' : ['none']}
    
    res = [k for k in dataf if 'reho' in k]
    param_grid    = {'subjsf': res, 'cl': ['linsvm'], 'prefs' : ['pearson', 'bhattacharyya'], 'prefs_thr': [90, 95], 'fs1' : ['none']}

    #cluster threshold
    clthresh = 0.95

    for j in list(ParameterGrid(param_grid)):

        subjsf    = j['subjsf']
        cl        = j['cl']
        prefs     = j['prefs']
        prefs_thr = j['prefs_thr']
        #fs1 = j['fs1']
        #fs2 = j['fs2']

        #of = os.path.join(wd, au.remove_ext(subjsf))
        of = os.path.join('/home/alexandre/Data/cobre/autoaq', au.remove_ext(os.path.basename(subjsf)))
        if prefs != 'none': of += '_' + prefs + '_' + str(prefs_thr)
        #if fs1   != 'none': of += '_' + fs1
        #if fs2   != 'none': of += '_' + fs2
        #of += '_' + cl

        #checking if output files exist
        out_file_name = of
        if atlas == 'MNI Structural Atlas':
            out_file_name += '_mni_struct'
        elif atlas == 'Harvard-Oxford Cortical Structural Atlas':
            out_file_name += '_harvox_cort'
        elif atlas == 'Harvard-Oxford Subcortical Structural Atlas':
            out_file_name += '_harvox_subcort'

        out_autoaq_txt = out_file_name + '_autoaq_presels.txt' #presels is the same for all classifiers

        of = of + '_' + cl
        svsqryf = of + '_autoaq_sup_vecs.txt'

        if not overwrite:
            if os.path.exists(out_autoaq_txt) and os.path.exists(svsqryf):
                au.log.info ("File " + out_autoaq_txt + " already exists, jumping to next.")
                continue

        au.log.info("Running " + prefs + " " + str(prefs_thr) + " " + str(j))

        #loading data file
        data_file = os.path.join(wd, 'features', os.path.basename(of) + '.pyshelf')
        au.log.info ('Loading ' + data_file)
        res = shelve.open(data_file)

        try:
            presels = res['presels']
            cv      = res['cv']
            sv      = res['importance']
        except:
            print('Problems reading ' + data_file)
            res.close()

        #Loading data within cv in order to unmask feature_importances_

        fidx   = dataf.index(subjsf)
        maskf  = masks[fidx]
        tmpltf = templates[fidx]

        mask, hdr, aff = au.get_nii_data(maskf)
        #maskidx = np.array(np.where(mask > 0))
        #hdr.set_data_dtype(np.dtype(np.float))

        tmplt  = nib.load(tmpltf).get_data()

        X = np.load(subjsf)
        y = np.loadtxt(labelsf)

        #getting localizations from data
        presels_vol, supvecs_vol, supvecs_done = get_localizations (X, y, cv, maskf, presels, sv)

        #standardizing presels
        from sklearn.preprocessing import MinMaxScaler
        mms = MinMaxScaler()
        presels_vol = mms.fit_transform(presels_vol)

        #binarizing presels
        #from sklearn.preprocessing import Binarizer
        #binr = Binarizer()
        #presels_vol = binr.fit_transform(presels_vol)

        print(out_autoaq_txt)

        #presels autoaq
        preselsf = os.path.join(wd, 'presels.nii.gz')
        au.save_nibabel (preselsf, presels_vol, aff, hdr)

        #resample to 2mm ?
        if maskf.find('3mm') > -1:
            au.log.info ('Resampling presels.')
            presels2f = os.path.join(wd, 'presels2mm.nii.gz')
            au.exec_command('3dresample -dxyz 2.0 2.0 2.0 -prefix ' + presels2f + ' -inset ' + preselsf)
            os.remove(preselsf)
            os.rename(presels2f, preselsf)

        #calling autoaq
        #http://brainder.org/tag/autoaq/
        prqryf = out_autoaq_txt
        call_autoaq (xd, preselsf, aq_thresh, atlas, prqryf)

        os.remove(preselsf)

        #sup_vecs/importances autoaq
        if supvecs_done:
            supvecsf = os.path.join(wd, 'sup_vecs.nii.gz')
            au.save_nibabel (supvecsf, supvecs_vol, aff, hdr)

            #resample to 2mm ?
            if maskf.find('3mm') > -1:
                au.log.info ('Resampling sup_vecs.')
                supvecs2f = os.path.join(wd, 'sup_vecs2mm.nii.gz')
                au.exec_command('3dresample -dxyz 2.0 2.0 2.0 -prefix ' + supvecs2f + ' -inset ' + supvecsf)
                os.remove(supvecsf)
                os.rename(supvecs2f, supvecsf)

            #svsqryf = of + '_autoaq_sup_vecs.txt'
            call_autoaq (xd, supvecsf, aq_thresh, atlas, svsqryf)

            os.remove(supvecsf)


##==============================================================================
'''
import sys
sys.path.append('/home/alexandre/Dropbox/Documents/work/cobre')
import do_cobre_classification as cs
from sklearn.grid_search import ParameterGrid

overwrite=True

#atlasquery atlases
atlases = []
#atlases.append('MNI Structural Atlas')
atlases.append('Harvard-Oxford Cortical Structural Atlas')
atlases.append('Harvard-Oxford Subcortical Structural Atlas')

temporal_filtering     = [True, False]
global_nuis_correction = [True, False]
preprocess_grid = {'temporal_filtering': [True, False], 'global_nuis_correction': [True, False], 'atlas': atlases}

for j in list(ParameterGrid(preprocess_grid)):
    cs.do_atlasquery(j['temporal_filtering'], j['global_nuis_correction'], overwrite, j['atlas'])
'''
def do_atlasquery (temporal_filtering=True, global_nuis_correction=True, overwrite=False, atlas='MNI Structural Atlas'):

    from sklearn.grid_search import ParameterGrid

    wd, xd, dd, labelsf, phenof, dataf, masks, dilmasks, templates, pipe = get_filepaths(temporal_filtering, global_nuis_correction)

    au.setup_logger(0, logfname=None)

    stratified, stddize, cvfolds, thrmethod, clf_methods, prefs_methods, prefs_thrs = get_experiment_parameters()
    #param_grid    = {'subjsf': dataf, 'cl': clf_methods, 'prefs' : prefs_methods, 'prefs_thr': prefs_thrs, 'fs1' : ['none']}
    
    res = [k for k in dataf if 'reho' in k]
    param_grid    = {'subjsf': res, 'cl': ['linsvm'], 'prefs' : ['pearson', 'bhattacharyya'], 'prefs_thr': [90, 95], 'fs1' : ['none']}

    #cluster threshold
    clthresh = 0.95

    for j in list(ParameterGrid(param_grid)):

        subjsf    = j['subjsf']
        cl        = j['cl']
        prefs     = j['prefs']
        prefs_thr = j['prefs_thr']
        #fs1 = j['fs1']
        #fs2 = j['fs2']

        #of = os.path.join(wd, au.remove_ext(subjsf))
        of = os.path.join('/home/alexandre/Data/cobre/autoaq', au.remove_ext(os.path.basename(subjsf)))
        if prefs != 'none': of += '_' + prefs + '_' + str(prefs_thr)
        #if fs1   != 'none': of += '_' + fs1
        #if fs2   != 'none': of += '_' + fs2
        #of += '_' + cl

        #checking if output files exist
        out_file_name = of
        if atlas == 'MNI Structural Atlas':
            out_file_name += '_mni_struct'
        elif atlas == 'Harvard-Oxford Cortical Structural Atlas':
            out_file_name += '_harvox_cort'
        elif atlas == 'Harvard-Oxford Subcortical Structural Atlas':
            out_file_name += '_harvox_subcort'

        out_atlasquerpy_txt = out_file_name + '_atlasquerpy.txt'
        out_atlasquery_txt = out_file_name + '_atlasquery.txt'

        if not overwrite:
            if os.path.exists(out_atlasquery_txt):
                au.log.info ("File " + out_atlasquery_txt + " already exists, jumping to next.")
                continue


        au.log.info("Running " + prefs + " " + str(prefs_thr) + " " + str(j))

        #loading data file
        data_file = os.path.join(wd, 'features', os.path.basename(of) + '.pyshelf')
        if not os.path.exists(data_file):
            oldf = data_file.replace('.pyshelf', '_linsvm.pyshelf')
            newf = data_file
            os.rename(oldf, newf)
            #au.log.error('File ' + data_file + ' does not exist. Skipping to the next.')
            #continue

        au.log.info ('Loading ' + data_file)
        res = shelve.open(data_file)

        try:
            presels = res['presels']
            cv      = res['cv']
            sv      = res['importance']
        except:
            print('Problems reading ' + data_file)
            res.close()

        #Loading data within cv in order to unmask feature_importances_

        fidx   = dataf.index(subjsf)
        maskf  = masks[fidx]
        tmpltf = templates[fidx]

        mask, hdr, aff = au.get_nii_data(maskf)
        #maskidx = np.array(np.where(mask > 0))
        #hdr.set_data_dtype(np.dtype(np.float))

        tmplt  = nib.load(tmpltf).get_data()

        X = np.load(subjsf)
        y = np.loadtxt(labelsf)

        #getting localizations from data
        presels_vol, supvecs_vol, supvecs_done = get_localizations (X, y, cv, maskf, presels, sv)

        #standardizing presels
        from sklearn.preprocessing import MinMaxScaler
        mms = MinMaxScaler()
        presels_vol = mms.fit_transform(presels_vol)

        #binarizing presels
        #from sklearn.preprocessing import Binarizer
        #binr = Binarizer()
        #presels_vol = binr.fit_transform(presels_vol)

        #saving presels nifti file
        preselsf = os.path.join(wd, 'presels.nii.gz')
        au.save_nibabel (preselsf, presels_vol, aff, hdr)

        #resample to 2mm ?
        if maskf.find('3mm') > -1:
            au.log.info ('Resampling presels.')
            presels2f = os.path.join(wd, 'presels2mm.nii.gz')
            au.exec_command('3dresample -dxyz 2.0 2.0 2.0 -prefix ' + presels2f + ' -inset ' + preselsf)
            os.remove(preselsf)
            os.rename(presels2f, preselsf)

        #binarise
        au.exec_command('fslmaths ' + preselsf + ' -bin ' + preselsf)

        #calling atlasquery
        #saving binary presels file first
        call_atlasquerpy(xd, preselsf, atlas, out_atlasquerpy_txt)
        #call_atlasquery(xd, preselsf, atlas, out_atlasquery_txt)

        os.remove(preselsf)


##==============================================================================
##Transforming numpy data format to matlab format
'''
import sys
sys.path.append('/home/alexandre/Dropbox/Documents/work/cobre')
import do_classification as cs
from sklearn.grid_search import ParameterGrid

insert_aal_idx = True

temporal_filtering     = [True, False]
global_nuis_correction = [True, False]
preprocess_grid = {'temporal_filtering': [True, False], 'global_nuis_correction': [True, False]}

for j in list(ParameterGrid(preprocess_grid)):
    cs.do_npy_to_mat(**j, insert_aal_idx)
'''
def do_npy_to_mat (temporal_filtering=True, global_nuis_correction=True, insert_aal_idx=False):
    import nibabel as nib
    import scipy.io as sio

    wd, xd, dd, labelsf, phenof, dataf, masks, dilmasks, templates, pipe = get_filepaths(temporal_filtering, global_nuis_correction)

    au.setup_logger(2, logfname=None)

    for fi in range(len(dataf)):

        subjsf = dataf[fi]
        maskf = masks[dataf.index(subjsf)]

        X = np.load(subjsf)
        y = np.loadtxt(labelsf)
        mask = nib.load(maskf).get_data()
        indices = np.where(mask > 0)

        #for COBRE ONLY, removing last two subjects:
        X = X[:-2,:]
        y = y[:-2]

        data = {'X': X.transpose(), 'y': y, 'mask': mask, 'mask_indices': indices}

        #should I insert AAL indices?
        if insert_aal_idx:
            if maskf.find('3mm') > -1:
                aalf = os.path.join(xd, 'aal_3mm.nii.gz')
            else:
                aalf = os.path.join(xd, 'aal_2mm.nii.gz')

            aal = nib.load(aalf).get_data()
            aal_indices = aal[indices]
            data.update({'aal_indices': aal_indices})

        of = au.remove_ext(subjsf) + '.mat'

        print('Saving ' + of)

        sio.savemat(of, data)

##==============================================================================
##Transforming numpy data format to matlab format, with prefeatsel
'''
import sys
sys.path.append('/home/alexandre/Dropbox/Documents/work/cobre')
import do_classification as cs

from sklearn.grid_search import ParameterGrid

temporal_filtering     = [True, False]
global_nuis_correction = [True, False]
preprocess_grid = {'temporal_filtering': [True, False], 'global_nuis_correction': [True, False]}

for j in list(ParameterGrid(preprocess_grid)):
    cs.do_npy_to_mat(**j)
'''
def do_npy_to_mat_with_prefeats():
    import nibabel as nib
    import scipy.io as sio

    temporal_filtering     = True
    global_nuis_correction = False
    wd, xd, dd, labelsf, phenof, dataf, masks, dilmasks, templates, pipe = get_filepaths(temporal_filtering, global_nuis_correction)

    verbose   = 2
    au.setup_logger(verbose, logfname=None)

    #pre feature selection processes
    prefs_methods = ['pearson', 'bhattacharyya', 'welcht']
    prefs_thrs = [80, 85, 90, 95, 99, 99.5]

    for fi in range(len(dataf)):

        subjsf = dataf[fi]
        maskf = masks[dataf.index(subjsf)]

        X = np.load(subjsf)
        y = np.loadtxt(labelsf)
        mask = nib.load(maskf).get_data()
        indices = np.where(mask > 0)

        #for COBRE ONLY, removing last two subjects:
        X = X[:-2,:]
        y = y[:-2]

        pearson = aizc.pre_featsel (X, y, 'pearson', 95)

        X_filt  = X[:, pearson > 0]

        data = {'X': X, 'y': y, 'mask': mask, 'mask_indices': indices, 'X_filt': X_filt, 'pearson': pearson}

        of = au.remove_ext(subjsf) + '.mat'

        sio.savemat(of, data)

#===============================================================================
# Creating localization images for Darya
def do_darya_localizations ():
    import nibabel as nib
    import scipy.io as sio

    sys.path.append('/home/alexandre/Dropbox/Documents/work/visualize_volume')
    import visualize_volume as vis

    import aizkolari_utils as au
    au.setup_logger()

    locd = '/home/alexandre/Dropbox/ELM 2013/ELM-2013-Darya/localization'

    wd, xd, dd, labelsf, phenof, dataf, masks, dilmasks, templates, pipe = get_filepaths()

    maskf          = dilmasks[0]
    mask, hdr, aff = au.get_nii_data(maskf)
    indices        = np.array(np.where(mask > 0))

    tmpltf         = templates[0]
    tmplt, _, _    = au.get_nii_data(tmpltf)

    flist = os.listdir(locd)
    flist = au.find(flist, '.mat')
    flist.sort()

    for f in flist:
        data = sio.loadmat(os.path.join(locd, f))

        name = au.remove_ext(f)

        if f.find('cobre') >= 0:
            p = data['pearson'].squeeze()
            locs = p > 0
            lidx = indices[:, locs].squeeze()

            if f.find('reho') >= 0:
                preho = p.copy()
            elif f.find('alff') >= 0:
                palff = p.copy()

        else:
            locs = data[name].squeeze()
            locs -= 1
            if f.find('pearson') >= 0:

                if f.find('reho') >= 0:
                    lidx = indices[:, preho > 0]
                elif f.find('alff') >= 0:
                    lidx = indices[:, palff > 0]

                lidx = lidx[:, locs]
            else:
                lidx = indices[:, locs].squeeze()

        locvol = np.zeros_like(mask, dtype=np.float)
        locvol[tuple(lidx)] = 1

        #save nifti volume
        au.save_nibabel (os.path.join(locd, name + '.nii.gz'), locvol, aff, hdr)

        #save image
        fig = vis.show_many_slices(tmplt, locvol, volaxis=1, vol2_colormap=plt.cm.autumn, figtitle=name)
        aizc.save_fig_to_png(fig, os.path.join(locd, name + '.png'))
        if os.access (os.path.join(xd,'innercrop'), os.X_OK):
            au.exec_command (os.path.join(xd,'innercrop') + ' -o white ' + fig2name + ' ' + fig2name)

#===============================================================================
##Transforming numpy data format to matlab format
'''
import sys
sys.path.append('/home/alexandre/Dropbox/Documents/work/cobre')
import do_classification as cs

from sklearn.grid_search import ParameterGrid

temporal_filtering     = [True, False]
global_nuis_correction = [True, False]
preprocess_grid = {'temporal_filtering': [True, False], 'global_nuis_correction': [True, False]}

for j in list(ParameterGrid(preprocess_grid)):
    cs.do_matlab_svm_bdopt(**j)
'''
def do_matlab_svm_bdopt (temporal_filtering=True, global_nuis_correction=True):
    import nibabel as nib
    import scipy.io as sio

    wd, xd, dd, labelsf, phenof, dataf, masks, dilmasks, templates, pipe = get_filepaths(temporal_filtering, global_nuis_correction)

    au.setup_logger(2, logfname=None)

    for fi in range(len(dataf)):

        myfile = au.remove_ext(subjsf) + '.mat'

        


