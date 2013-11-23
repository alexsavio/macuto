#!/usr/bin/python

import os
import re
import sys
import argparse
import subprocess
import logging as log
import numpy as np
import nibabel as nib
import shelve

from sklearn.grid_search import ParameterGrid

sys.path.append('/home/alexandre/Dropbox/Documents/phd/work/aizkolari')
import aizkolari_utils as au
import aizkolari_classification as aizc

from do_namic_featsets import get_filepaths
from aizkolari_classification import Result

from IPython.core.debugger import Tracer; debug_here = Tracer()

#-------------------------------------------------------------------------------
def set_parser():
    parser = argparse.ArgumentParser(description='Script for multi-label classification on OASIS, with many feature extraction options')

    clfmethods   = ['cart', 'gmm', 'rf', 'svm', 'sgd', 'linsvm', 'percep']
    prefsmethods = ['none', 'pearson', 'bhattacharyya', 'welcht']
    fsmethods    = ['rfe', 'rfecv', 'univariate', 'fdr', 'fpr', 'extratrees', 'pca', 'rpca', 'lda'] #svmweights
    thrmethods   = ['robust', 'percentile', 'rank']

    parser.add_argument('-i', '--in', dest='infile', required=True, 
            help='''can be either a text files with a list with the subject files for the analysis, where each line: <class_label>,<subject_file>.\n
                    or a .npy file with a matrix with the data NxM, where N is the number of subjects, a class label file must be given with the -l argument.
                 ''')

    parser.add_argument('-o', '--out',    dest='outfile', required=True, help='Python shelve output file name preffix.')

    parser.add_argument('-l', '--labels', dest='labels', required=False, help='Text file with the class labels for the subjects, should be in the same order as the input file. One label per line.')

    parser.add_argument('-d', '--datadir', dest='datadir', required=False, default='', help='folder path where the subjects are, if the absolute path is not included in the subjects list file.')

    parser.add_argument('-m', '--mask',  dest='mask', required=True, help='Mask file to extract feature voxels, any voxel with values > 0 will be included in the extraction.')

    parser.add_argument('--cvfold',      dest='cvfold',      default='10', choices=['10', 'loo'], required=False, help='Cross-validation folding method: stratified 10-fold or leave-one-out.')
    parser.add_argument('-e', '--estim', dest='clfmethod',   default='svm', choices=clfmethods, required=False, help='classifier type')

    parser.add_argument('--prefsmethod', dest='prefsmethod', default='none', choices=prefsmethods, required=False, help='Pre-feature selection method')

    parser.add_argument('--prefsthr',    dest='prefsthr', default=95, type=int, required=False, help='Pre-feature selection method threshold [0-100]')

    parser.add_argument('--fsmethod1',   dest='fsmethod1', default='rfe', choices=fsmethods, required=False, help='First feature selection method')

    parser.add_argument('--fsmethod2',   dest='fsmethod2', default='None', choices=fsmethods, required=False, help='Second feature selection method')

    parser.add_argument('--thrmethod',   dest='thrmethod', default='robust', choices=thrmethods, required=False, 
            help='''Threshold method used after the pre feature selection step. \n
                    This option does not affect the prefsthr option.
                    Robustness: robust >> percentile >> rank
                 ''')

    parser.add_argument('--scale',       dest='scale', default=False, action='store_true', required=False, help='This option will enable Range scaling of the training data.')

    parser.add_argument('--scale_min',   dest='scale_min', default=-1, type=int, required=False, help='Minimum value for the new scale range.')

    parser.add_argument('--scale_max',   dest='scale_max', default=1, type=int, required=False, help='Maximum value for the new scale range.')

    parser.add_argument('--ncpus',       dest='ncpus', required=False, type=int, default=1, help='number of cpus used for parallelized grid search')

    parser.add_argument('-v', '--verbosity', dest='verbosity', required=False, type=int, default=2, help='Verbosity level: Integer where 0 for Errors, 1 for Input/Output, 2 for Progression reports')


    return parser

#-------------------------------------------------------------------------------
def do_classification (subjsf, labelsf, outfile, datadir, maskf, clfmethod, fsmethod1, fsmethod2, prefsmethod, prefsthr, cvmethod, thrmethod, stratified, stddize, n_cpus):
    #first list of variables to be saved
    shelfnames = ['subjsf', 'maskf', 'datadir', 'prefsmethod', 'prefsthr', 'clfmethod', 'fsmethod1', 'fsmethod2', 'stddize']

    scores = None
    if au.get_extension(subjsf) != '.npy':
        X, y, scores, imgsiz, mask, indices = aizc.load_data (subjsf, datadir, maskf, labelsf)
    else:
        X = np.load(subjsf)
        y = np.loadtxt(labelsf)
        mask = nib.load(maskf).get_data()
        indices = np.where(mask > 0)

    preds, probs, best_pars, presels, cv, importance, scores, y, truth = aizc.extract_classify (X, y, scores, prefsmethod, prefsthr, fsmethod1, fsmethod2, clfmethod, cvmethod, stratified, stddize, thrmethod, n_cpus)

    #save results with pickle
    au.log.info('Saving results in ' + outfile)

    #extend list of variables to be saved
    shelfnames.extend([ 'y', 'mask', 'indices', 'cv', 'importance', 'preds', 'probs', 'scores', 'best_pars', 'truth', 'presels'])

    shelfvars = []
    for v in shelfnames:
        shelfvars.append(eval(v))

    au.shelve_varlist(outfile + '.pyshelf', shelfnames, shelfvars)

    return cv, truth, preds, probs, presels

#-------------------------------------------------------------------------------
## START MAIN
#-------------------------------------------------------------------------------
def main(argv=None):

    #parsing arguments
    parser = set_parser()

    try:
        args = parser.parse_args ()
    except argparse.ArgumentError, exc:
        print (exc.message + '\n' + exc.argument)
        parser.error(str(msg))
        return 0

    subjsf      = args.infile.strip     ()
    labelsf     = args.labels.strip     ()
    outfile     = args.outfile.strip    ()
    datadir     = args.datadir.strip    ()
    maskf       = args.mask.strip       ()
    clfmethod   = args.clfmethod.strip  ()
    fsmethod1   = args.fsmethod1.strip  ()
    fsmethod2   = args.fsmethod2.strip  ()
    prefsmethod = args.prefsmethod.strip()
    prefsthr    = args.prefsthr
    thrmethod   = args.thrmethod.strip()

    cvfold     = args.cvfold.strip()
    stddize    = args.scale
    n_cpus     = args.ncpus

    verbose    = args.verbosity

    stratified = True

    #logging config
    au.setup_logger(verbose)

    return do_classification (subjsf, labelsf, outfile, datadir, maskf, clfmethod, fsmethod1, fsmethod2, prefsmethod, prefsthr, cvfolds, thrmethod, stratified, stddize, n_cpus)


################################################################################
#MAIN
if __name__ == "__main__":
    sys.exit(main())

##==============================================================================
def get_experiment_parameters ():

    stratified = True
    stddize    = True
    cvfolds    = 'loo'
    thrmethod  = 'robust'

    #EXPERIMENT 1
    #pre feature selection processes
    clf_methods   = ['linsvm', 'rbfsvm', 'rf']
    prefs_methods = ['pearson', 'bhattacharyya', 'welcht']
    prefs_thrs    = [80, 85, 90, 95, 99, 99.5]

    ##EXPERIMENT 2
    ##creating grid of experiments parameters
    #clf_methods    = ['cart', 'rf', 'svm', 'sgd', 'linsvm', 'percep']
    #clf_methods    = ['linsvm', 'svm', 'rf']
    #fs_methods     = ['rfe', 'rfecv', 'univariate', 'fdr', 'fpr', 'extratrees', 'pca', 'rpca', 'lda'] #svmweights
    #few_fs_methods = ['univariate', 'fdr', 'fpr', 'extratrees', 'rfe']
    #param_grid     = {'subjsf': dataf, 'cl': clf_methods, 'prefs' : ['none'], 'prefs_thr': ['none'], 'fs1' : few_fs_methods}

    fs2 = 'none'

    return stratified, stddize, cvfolds, thrmethod, clf_methods, prefs_methods, prefs_thrs

#-------------------------------------------------------------------------------
def do_paramgrid (wd, masks, dataf, labelsf, param_grid, thrmethod, stratified, stddize, cvfolds, n_cpus):

    results = {}
    for j in list(ParameterGrid(param_grid)):

        subjsf    = j['subjsf']
        cl        = j['cl']
        prefs     = j['prefs']
        prefs_thr = j['prefs_thr']
        fs1       = j['fs1']
        fs2       = j['fs2']

        maskf = masks[dataf.index(subjsf)]

        of = os.path.join(wd, au.remove_ext(j['subjsf']))
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
sys.path.append('/home/alexandre/Dropbox/Documents/phd/work/namic_brainmulti')

import do_namic_classification as nc
results = nc.do_experiment(4)

nc.plot_all_results (results)
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
        results[str(j)] = do_paramgrid(wd, masks, dataf, labelsf, param_grid, thrmethod, stratified, stddize, cvfolds, n_cpus)

        #plot_results (results[str(j)], wd, masks, dataf, param_grid, stratified, stddize, cvfolds)

    return results

#==============================================================================
def subplot_this(plt, ax, xlabel, ylabel, prefs_thrs, yvals, yvariances, yrange, c,  clor='k', mark='o', sty='_'):
    line = ax.errorbar(prefs_thrs, yvals, yerr=yvariances, color=clor, marker=mark, ls=sty, label=c, lw=2, elinewidth=2, capsize=5)
    ax.set_xlabel(xlabel, size='x-large')
    ax.set_ylabel(ylabel, labelpad=10, size='x-large')
    ax.set_yticks(yrange)
    ax.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75')
    ax.grid(which='minor', axis='x', linewidth=0.25, linestyle='-', color='0.75')
    ax.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75')
    ax.grid(which='minor', axis='y', linewidth=0.25, linestyle='-', color='0.75')
    plt.xticks(prefs_thrs, rotation='vertical')
    ax.legend(loc=3)

##==============================================================================
def plot_all_results(results):
    temporal_filtering     = [True, False]
    global_nuis_correction = [True, False]
    preprocess_grid = {'temporal_filtering': [True, False], 'global_nuis_correction': [True, False]}

    stratified, stddize, cvfolds, thrmethod, clf_methods, prefs_methods, prefs_thrs = get_experiment_parameters()

    for j in list(ParameterGrid(preprocess_grid)):

        wd, xd, dd, labelsf, phenof, dataf, masks, dilmasks, templates, pipe = get_filepaths(**j)

        print str(j)
        print 'Doing pipeline ' + pipe + '\n'

        plot_results(results[str(j)], wd, masks, dataf, prefs_methods, prefs_thrs, clf_methods)

##==============================================================================
def plot_results (results, wd, masks, subjsf, prefs_methods, prefs_thrs, clf_methods):

    ##SHOW FINAL RESULTS
    import pylab as plt

    datatype = [
    'ALFF',
    'fALFF',
    'Regional Homogeneity',
    'VMHC',
    'VMHC statistical maps',
    ]

    yrange = np.arange(0.5, 1.0, 0.1)

    #colors = ['ro-', 'gx-', 'bs-']
    colors = ['r', 'g', 'b', 'k', 'y', 'm']
    styles = ['-', '--', ':', '_']
    markrs = ['D', 'o', 'v', '+', 's', 'x']

    for fi, f in enumerate(subjsf):

        f = subjsf[fi]
        print f
        resf = results[f]
        print len(resf)

        for p in prefs_methods:
            print p
            resfp = aizc.filter_objlist (resf, 'prefs', p)

            fig = plt.figure(p + '_' + os.path.basename(f))

            i = 0
            for c in clf_methods:
                print c
                resfpc = aizc.filter_objlist (resfp, 'cl', c)

                clor = colors[i]
                mark = markrs[i]
                sty  = styles[i]
                i += 1

                # getting accuracy, spec, sens
                maccs, msens, mspec, mprec, mfone, mrauc = [], [], [], [], [], []
                vaccs, vsens, vspec, vprec, vfone, vrauc = [], [], [], [], [], []
                for t in prefs_thrs:
                    resfpct = aizc.filter_objlist (resfpc, 'prefs_thr', t)[0]

                    #metrics[i, :] = np.array([acc, sens, spec, prec, f1, roc_auc])
                    metrs = np.array(resfpct.metrics)

                    means = metrs.mean(axis=0)
                    varis = metrs.var (axis=0)

                    #get mean accuracy, sens and spec
                    maccs.append(means[0])
                    msens.append(means[1])
                    mspec.append(means[2])
                    mprec.append(means[3])
                    mfone.append(means[4])
                    mrauc.append(means[5])

                    #get var accuracy, sens and spec
                    vaccs.append(varis[0])
                    vsens.append(varis[1])
                    vspec.append(varis[2])
                    vprec.append(varis[3])
                    vfone.append(varis[4])
                    vrauc.append(varis[5])

                xlabel = 'threshold'

                ylabel = 'accuracy'
                ax     = plt.subplot(2,3,1)
                subplot_this(plt, ax, xlabel, ylabel, prefs_thrs, maccs, vaccs, yrange, c, clor, mark, sty)

                ylabel = 'sensitivity'
                ax     = plt.subplot(2,3,2)
                subplot_this(plt, ax, xlabel, ylabel, prefs_thrs, msens, vsens, yrange, c, clor, mark, sty)

                figure_title = str.upper(p[0]) + p[1:] + ' on ' + datatype[fi]
                plt.text(0.5, 1.08, figure_title, horizontalalignment='center', fontsize=20, transform = ax.transAxes, fontname='Ubuntu')
                #plt.title (figure_title)

                ylabel = 'specificity'
                ax     = plt.subplot(2,3,3)
                subplot_this(plt, ax, xlabel, ylabel, prefs_thrs, mspec, vspec, yrange, c, clor, mark, sty)

                ylabel = 'precision'
                ax     = plt.subplot(2,3,4)
                subplot_this(plt, ax, xlabel, ylabel, prefs_thrs, mprec, vprec, yrange, c, clor, mark, sty)

                ylabel = 'F1-score'
                ax     = plt.subplot(2,3,5)
                subplot_this(plt, ax, xlabel, ylabel, prefs_thrs, mfone, vfone, yrange, c, clor, mark, sty)

                ylabel = 'ROC AUC'
                ax     = plt.subplot(2,3,6)
                subplot_this(plt, ax, xlabel, ylabel, prefs_thrs, mrauc, vrauc, yrange, c, clor, mark, sty)

            #fig.show()
            #raw_input("Press Enter to continue...")
            plot_fname = os.path.join(wd, p + '_' + os.path.basename(f) + '.png')
            aizc.save_fig_to_png(fig, plot_fname, 'white')
            plt.close()


##==============================================================================
##execution script
'''
import sys
sys.path.append('/home/alexandre/Dropbox/Documents/phd/work/namic_brainmulti')

import do_namic_classification as cb
results = cb.do_experiment(2)

sum_tabs = cb.print_all_summary_tables (results)

'''
def print_all_summary_tables(results):
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
                means, varis = [], []
                m_idx = 0 #metric index
                for t in prefs_thrs:
                    resfpct = aizc.filter_objlist (resfpc, 'prefs_thr', t)[0]

                    #metrics[i, :] = np.array([acc, sens, spec, prec, f1, roc_auc])
                    metrs = np.array(resfpct.metrics)

                    means.append(metrs.mean(axis=0))
                    varis.append(metrs.var (axis=0))

                max_means = np.max(means, axis=1)
                max_varis = np.max(varis, axis=1)

                row = "%.2f (%.2f)" % (max_means[m_idx], max_varis[m_idx])
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
sys.path.append('/home/alexandre/Dropbox/Documents/phd/work/namic_brainmulti')
import do_namic_classification as nc
from sklearn.grid_search import ParameterGrid

volaxis=2
save_fig=True
overwrite=True

temporal_filtering     = [True, False]
global_nuis_correction = [True, False]
preprocess_grid = {'temporal_filtering': [True, False], 'global_nuis_correction': [True, False]}

for j in list(ParameterGrid(preprocess_grid)):
    cb.do_localization_masks(j['temporal_filtering'], j['global_nuis_correction'], 
                        save_fig=save_fig, volaxis=volaxis, overwrite=overwrite)
'''
def do_localization_masks (temporal_filtering=True, global_nuis_correction=True, 
                           save_fig=True, volaxis=0, overwrite=False):

    if save_fig:
        import matplotlib
        matplotlib.use('AGG')

    import pylab as plt
    from sklearn.grid_search import ParameterGrid

    sys.path.append('/home/alexandre/Dropbox/Documents/phd/work/visualize_volume')
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

        of = os.path.join(wd, au.remove_ext(j['subjsf']))
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

##==============================================================================
def call_autoaq (xd, maskf, thresh, atlas, outfile):
    if os.access (os.path.join(xd,'autoaq'), os.X_OK):
        comm = os.path.join(xd,'autoaq') + ' -i ' + maskf + ' -t ' + str(thresh) + ' -a ' + '"' + atlas + '"' + ' -o ' + outfile
        print ('Calling: ' + comm)
        au.exec_command (comm)

##==============================================================================
'''
import sys
sys.path.append('/home/alexandre/Dropbox/Documents/phd/work/namic_brainmulti')
import do_namic_classification as cs
from sklearn.grid_search import ParameterGrid

overwrite=False

temporal_filtering     = [True, False]
global_nuis_correction = [True, False]
preprocess_grid = {'temporal_filtering': [True, False], 'global_nuis_correction': [True, False]}

for j in list(ParameterGrid(preprocess_grid)):
    cs.do_atlasquery(j['temporal_filtering'], j['global_nuis_correction'], overwrite=overwrite)
'''
def do_atlasquery (temporal_filtering=True, global_nuis_correction=True, overwrite=False):

    from sklearn.grid_search import ParameterGrid

    wd, xd, dd, labelsf, phenof, dataf, masks, dilmasks, templates, pipe = get_filepaths(temporal_filtering, global_nuis_correction)

    au.setup_logger(2, logfname=None)

    stratified, stddize, cvfolds, thrmethod, clf_methods, prefs_methods, prefs_thrs = get_experiment_parameters()
    param_grid    = {'subjsf': dataf, 'cl': clf_methods, 'prefs' : prefs_methods, 'prefs_thr': prefs_thrs, 'fs1' : ['none']}

    #cluster threshold
    clthresh = 0.95

    #atlasquery atlases
    #atlas = 'MNI Structural Atlas',
    atlas = 'Harvard-Oxford Cortical Structural Atlas'

    for j in list(ParameterGrid(param_grid)):

        subjsf    = j['subjsf']
        cl        = j['cl']
        prefs     = j['prefs']
        prefs_thr = j['prefs_thr']
        #fs1 = j['fs1']
        #fs2 = j['fs2']

        of = os.path.join(wd, au.remove_ext(subjsf))
        #of = os.path.join('/home/alexandre/Data/namic_brainmultimodality/autoaq', au.remove_ext(os.path.basename(subjsf)))
        if prefs != 'none': of += '_' + prefs + '_' + str(prefs_thr)
        #if fs1   != 'none': of += '_' + fs1
        #if fs2   != 'none': of += '_' + fs2
        #of += '_' + cl

        #checking if output files exist
        out_autoaq_txt = of + '_autoaq_presels.txt' #presels is the same for all classifiers

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
        finally:
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

        #presels autoaq
        if not os.path.exists(out_autoaq_txt):
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
            call_autoaq (xd, preselsf, clthresh, atlas, prqryf)

            os.remove(preselsf)

        #sup_vecs/importances autoaq
        if not os.path.exists(svsqryf):
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
                call_autoaq (xd, supvecsf, clthresh, atlas, svsqryf)

                os.remove(supvecsf)


##==============================================================================
##Transforming numpy data format to matlab format
'''
import sys
sys.path.append('/home/alexandre/Dropbox/Documents/phd/work/namic_brainmulti')
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
sys.path.append('/home/alexandre/Dropbox/Documents/phd/work/namic_brainmulti')
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

    sys.path.append('/home/alexandre/Dropbox/Documents/phd/work/visualize_volume')
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

        if f.find('namic') >= 0:
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
sys.path.append('/home/alexandre/Dropbox/Documents/phd/work/namic_brainmulti')
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

        


