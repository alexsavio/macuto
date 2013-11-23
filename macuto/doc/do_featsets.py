#!/usr/bin/python

import os
import re
import sys
import argparse
import numpy as np
import nibabel as nib
import pickle

import scipy.stats as stats
import scipy.io as sio

#feature selection
from sklearn.feature_selection import f_classif
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectFdr
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import RFECV
from sklearn.cross_validation import KFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import zero_one

#data preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

#other decompositions
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
from sklearn.lda import LDA

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import LeaveOneOut, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline, FeatureUnion

from IPython.core.debugger import Tracer; debug_here = Tracer()

sys.path.append('/home/alexandre/Dropbox/Documents/phd/work/aizkolari')
import aizkolari_utils as au


'''
import os
import sys

sys.path.append('/home/alexandre/Dropbox/Documents/phd/work/aizkolari')
import aizkolari_utils as au

sys.path.append('/home/alexandre/Dropbox/Documents/phd/work/oasis_aal')
from do_aal_featsets import *

hn = au.get_hostname()
if hn == 'azteca':
    wd = '/media/data/oasis_aal'
    roisdir = ''
    outd    = '/media/data/oasis_aal'
elif hn == 'corsair':
    wd = '/media/alexandre/alextenso/work/oasis_svm'
    roisdir = '/scratch/oasis_aal/aal_rois'
    outd    = '/scratch/oasis_aal'
elif hn == 'hpmed':
    wd = '/media/alexandre/toshiba/work/oasis_svm'
    roisdir = ''
    outd    = '/media/alexandre/toshiba/work/oasis_svm'

verbose   = 2
au.setup_logger(verbose, logfname=None)

otype = '.npy'

smoothmm = 2

roilabsf  = '/home/alexandre/Dropbox/Documents/phd/work/oasis_aal/aal_allvalues.txt'

#fsmethods  = ['none', 'stats']
fsmethods  = ['stats']
featstypes = ['jacs','smoothmodgm','modulatedgm','geodan','norms','trace']

for feats in featstypes:
    datadir = os.path.join (wd, feats)

    subjlstf = os.path.join (wd, feats + '_lst')

    for fsmethod in fsmethods:
        outfsdir  = 'oasis_' + feats
        if smoothmm > 0:
            outfsdir += '_' + str(smoothmm) + 'mm'
        outfsdir += '_' + fsmethod

        outdir  = os.path.join(outd, outfsdir)
        print ('Creating ' + outdir)
        main_do (datadir, subjlstf, smoothmm, feats, outdir, roisdir, roilabsf, fsmethod, otype)

'''

#-------------------------------------------------------------------------------
def set_parser():
    fsmethods    = ['none', 'stats', 'hist3d']  # ONLY UNSUPERVISED FEATURE EXTRACTION METHODS HERE, NOT 'rfe', 'rfecv', 'univariate', 'fdr', 'fpr', 'extratrees', 'pca', 'rpca', 'lda'] #svmweights
    feats        = ['jacs','smoothmodgm','modulatedgm', 'geodan', 'norms', 'trace']
    outs         = ['.npy','.mat']

    parser = argparse.ArgumentParser(description='OASIS AAL classification experiment.')
    parser.add_argument('-s', '--subjlstf',  dest='subjlstf', default='', required=True,   help='list file with the subjects for the analysis. Each line: <class_label>,<subject_file>')
    parser.add_argument('-d', '--datadir',   dest='datadir',  default='', required=True,   help='data directory path')
    parser.add_argument('-r', '--roisdir',   dest='roisdir',  default='', required=True,   help='rois directory path')
    parser.add_argument('-o', '--outdir',    dest='outdir',   default='', required=False,  help='output data directory path')
    parser.add_argument('-l', '--roilabsf',  dest='roilabsf', default='', required=True,   help='path of file with ROI labels-values table, where each line has: ROIlabel ROIvalue')
    parser.add_argument('-f', '--feats',     dest='feats',    default='jacs', choices=feats, required=False, help='deformation measure type')
    parser.add_argument(      '--otype',     dest='otype',    default='.npy', choices=outs,  required=False, help='output file type.')

    parser.add_argument('--bins',            dest='bins',     default=10, required=False, type=int, help='Number of bins in each dimension of the volume to calculate the histogram features.')
    parser.add_argument('--smoothmm',        dest='smoothmm', default=0, required=False, type=int, help='Size of a Gaussian smooth filter for each image before selecting features.')
    parser.add_argument('--fsmethod',        dest='fsmethod', default='rfe', choices=fsmethods, required=False, help='Feature selection method')

    parser.add_argument('-v', '--verbosity', dest='verbosity', required=False, type=int, default=2, help='Verbosity level: Integer where 0 for Errors, 1 for Input/Output, 2 for Progression reports')

    return parser

#-------------------------------------------------------------------------------
def get_aal_info(aal_data, roi_idx):
   return aal_data[aal_data[:,3] == str(roi_idx)].flatten()

#-------------------------------------------------------------------------------
def list_filter (list, filter):
    return [ (l) for l in list if filter(l) ]

#-------------------------------------------------------------------------------
def dir_search (regex, wd='.'):
    ls = os.listdir(wd)

    filt = re.compile(regex).search
    return list_filter(ls, filt)

#-------------------------------------------------------------------------------
def dir_match (regex, wd='.'):
    ls = os.listdir(wd)

    filt = re.compile(regex).match
    return list_filter(ls, filt)

#-------------------------------------------------------------------------------
def list_match (regex, list):
    filt = re.compile(regex).match
    return list_filter(list, filt)

#-------------------------------------------------------------------------------
def list_search (regex, list):
    filt = re.compile(regex).search
    return list_filter(list, filt)

#-------------------------------------------------------------------------------
def shelve_vars (ofname, varlist):
   mashelf = shelve.open(ofname, 'n')

   for key in varlist:
      try:
         mashelf[key] = globals()[key]
      except:
         log.error('ERROR shelving: {0}'.format(key))

   mashelf.close()

#-------------------------------------------------------------------------------
def append_to_keys (mydict, preffix):
    return {preffix + str(key) : (transform(value) if isinstance(value, dict) else value) for key, value in mydict.items()}

#-------------------------------------------------------------------------------
def append_to_list (mylist, preffix):
    return list({preffix + str(item) for item in mylist})

#-------------------------------------------------------------------------------
def join_path_to_filelist (path, mylist):
    return list({os.path.join(path, str(item)) for item in mylist})

#-------------------------------------------------------------------------------
def save_feats_file (feats, otype, outfname):
    if   otype == '.npy': np.save(outfname + '.npy', feats)
    elif otype == '.mat': sio.savemat(outfname + '.mat', dict(feats = feats))

#-------------------------------------------------------------------------------
def get_fsmethod (fsmethod, n_feats, n_subjs):
    n_jobs = 2

    #Feature selection procedures
    from sklearn.cross_validation import StratifiedKFold
                                #http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
    fsmethods = { 'rfe'       : RFE(estimator=SVC(kernel="linear", C=1), step=0.05, n_features_to_select=2),
                                #http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
                  'rfecv'     : RFECV(estimator=SVC(kernel="linear"), step=0.05, loss_func=zero_one), #cv=3, default; cv=StratifiedKFold(n_subjs, 3)
                                #Univariate Feature selection: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html
                  'univariate': SelectPercentile(f_classif, percentile=5),
                                #http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFpr.html
                  'fpr'       : SelectFpr (f_classif, alpha=0.05),
                                #http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFdr.html
                  'fdr'       : SelectFdr (f_classif, alpha=0.05),
                                #http://scikit-learn.org/stable/modules/feature_selection.html
                  'extratrees': ExtraTreesClassifier(n_estimators=50, max_features='auto', compute_importances=True, n_jobs=n_jobs, random_state=0),

                  'pca'       : PCA(n_components='mle'),
                  'rpca'      : RandomizedPCA(random_state=0),
                  'lda'       : LDA(),
    }

    #feature selection parameter values for grid search
    max_feats = ['auto']
    if n_feats < 10:
        feats_to_sel = range(2, n_feats, 2)
        n_comps = range(1, n_feats, 2)
    else:
        feats_to_sel = range(2, 20, 4)
        n_comps = range(1, 30, 4)
    max_feats.extend(feats_to_sel)

    fsgrid =    { 'rfe'       : dict(estimator__C = [0.1, 1, 10], n_features_to_select = feats_to_sel),
                  'rfecv'     : dict(estimator__C = [0.1, 1, 10]),
                  'univariate': dict(percentile = [1, 3, 5, 10]),
                  'fpr'       : dict(alpha = [1, 3, 5, 10]),
                  'fdr'       : dict(alpha = [1, 3, 5, 10]),
                  'extratrees': dict(n_estimators = [1, 3, 5, 10, 30, 50], max_features = max_feats),
                  'pca'       : dict(n_components = n_comps.extend(['mle']), whiten = [True, False]),
                  'rpca'      : dict(n_components = n_comps, iterated_power = [3, 4, 5], whiten = [True, False]),
                  'lda'       : dict(n_components = n_comps)
    }

    return fsmethods[fsmethod], fsgrid[fsmethod]

#-------------------------------------------------------------------------------
def create_rois_mask (roilst, roiflst):
    au.log.info('Creating all ROIs common mask for data reading')

    shape = nib.load(roiflst[0]).shape
    mask  = np.zeros(shape)

    #create space for all features and read from subjects
    for roi in roilst:
        try:
            roif   = list_search('_' + roi + '.', roiflst)[0]
            roivol = nib.load(roif).get_data()
            mask += roivol
        except:
            debug_here()

    return mask > 0

#-------------------------------------------------------------------------------
def parse_subjects_list (fname, datadir=''):
    labels = []
    subjs  = []

    if datadir:
        datadir += os.path.sep

    try:
        f = open(fname, 'r')
        for s in f:
            line = s.strip().split(',')
            labels.append(np.float(line[0]))
            subjf = line[1].strip()
            if not os.path.isabs(subjf):
                subjs.append (datadir + subjf)
            else:
                subjs.append (subjf)
        f.close()

    except:
        au.log.error( "Unexpected error: ", sys.exc_info()[0] )
        debug_here()
        sys.exit(-1)

    return [labels, subjs]

#-------------------------------------------------------------------------------
def smooth_volume(imf, smoothmm):
    from nipype.interfaces.fsl.maths import IsotropicSmooth

    if smoothmm > 0:
        omf = imf + '_smooth' + str(smoothmm) + 'mm.nii.gz'

        isosmooth = IsotropicSmooth()
        isosmooth.inputs.in_file  = imf
        isosmooth.inputs.fwhm     = smoothmm
        isosmooth.inputs.out_file = omf
        isosmooth.run()

        data = nib.load(omf).get_data()
        os.remove(omf)
    else:
        data = nib.load(imf).get_data()

    return data

#-------------------------------------------------------------------------------
def load_data (subjsf, datadir, msk, smoothmm=0):

    #loading mask
    nvox    = np.sum  (msk > 0)
    indices = np.where(msk > 0)

    #reading subjects list
    [scores, subjs] = parse_subjects_list (subjsf, datadir)
    scores = np.array(scores)

    imgsiz = nib.load(subjs[0]).shape
    dtype  = nib.load(subjs[0]).get_data_dtype()
    nsubjs = len(subjs)

    #checking mask and first subject dimensions match
    if imgsiz != msk.shape:
        au.log.error ('Subject image and mask dimensions should coincide.')
        exit(1)

    #relabeling scores to integers, if needed
    if not np.all(scores.astype(np.int) == scores):
    #    unis = np.unique(scores)
    #    scs  = np.zeros (scores.shape, dtype=int)
    #    for k in np.arange(len(unis)):
    #        scs[scores == unis[k]] = k
    #    y = scs.copy()
        le = LabelEncoder()
        le.fit(scores)
        y = le.transform(scores)
    else:
        y = scores.copy()

    #loading data
    au.log.info ('Loading data...')
    X = np.zeros((nsubjs, nvox), dtype=dtype)
    for f in np.arange(nsubjs):
        imf = subjs[f]

        au.log.info('Reading ' + imf)

        if (smoothmm > 0):
            img = smooth_volume(imf, smoothmm)
        else:
            img = nib.load(imf).get_data()

        X[f,:] = img[msk > 0]

    return X, y, scores, imgsiz, indices 

#-------------------------------------------------------------------------------
def calculate_stats (data):
    n_subjs = data.shape[0]

    feats  = np.zeros((n_subjs, 7))

    feats[:,0] = fs.max (axis=1)
    feats[:,1] = fs.min (axis=1)
    feats[:,2] = fs.mean(axis=1)
    feats[:,3] = fs.var (axis=1)
    feats[:,4] = np.median      (fs, axis=1)
    feats[:,5] = stats.kurtosis (fs, axis=1)
    feats[:,6] = stats.skew     (fs, axis=1)

    return feats

#-------------------------------------------------------------------------------
def calculate_hist3d (data, bins):
    n_subjs = data.shape[0]

    feats = np.zeros((n_subjs, bins*bins*bins))

    for s in np.arange(n_subjs):
        H, edges = np.histogramdd(data[s,], bins = (bins, bins, bins))
        feats[s,:] = H.flatten()

    return feats

#-------------------------------------------------------------------------------
def create_feature_sets (fsmethod, fsgrid, data, msk, y, outdir, outbasename, otype):
    n_subjs = data.shape[0]

    aalinfo = np.loadtxt (roilabsf, dtype=str)

    np.savetxt (os.path.join(outdir, outbasename + '_labels.txt'), y, fmt="%.2f")

    outfname = os.path.join(outdir, outbasename)
    au.log.info('Creating ' + outfname)

    fs = data[:, msk > 0]

    if fsmethod == 'stats':
        feats = calculate_stats (fs)

    elif fsmethod == 'hist3d':
        feats = calculate_hist3d (fs)

    elif fsmethod == 'none':
        feats = fs

    #save file
    save_feats_file (feats, otype, outfname)

#-------------------------------------------------------------------------------
def create_feature_sets (fsmethod, fsgrid, data, msk, y, roilst, roiflst, roilabsf, outdir, outbasename, otype):

    n_subjs = data.shape[0]

    aalinfo = np.loadtxt (roilabsf, dtype=str)

    np.savetxt (os.path.join(outdir, outbasename + '_labels.txt'), y, fmt="%.2f")

    for roi in roilst:

        outfname = os.path.join(outdir, outbasename + '_' + roi)
        au.log.info('Creating ' + outfname)

        roif   = list_search('_' + roi + '.', roiflst)[0]

        aalidx = [i for i, x in enumerate(aalinfo[:,0]) if x == roi]
        aalrow = aalinfo[aalidx,:]

        #load roi
        roivol = nib.load(roif).get_data()

        roivol = roivol[msk > 0]

        if fsmethod == 'stats':
            feats  = np.zeros((n_subjs, 7))

            fs = data[:, roivol > 0]

            feats[:,0] = fs.max (axis=1)
            feats[:,1] = fs.min (axis=1)
            feats[:,2] = fs.mean(axis=1)
            feats[:,3] = fs.var (axis=1)
            feats[:,4] = np.median(fs, axis=1)
            feats[:,5] = stats.kurtosis (fs, axis=1)
            feats[:,6] = stats.skew     (fs, axis=1)

        elif fsmethod == 'none':
            feats = data[:, roivol > 0]

        #save file
        save_feats_file (feats, otype, outfname)


#-------------------------------------------------------------------------------
def main_do  (datadir, subjlstf, bins, smoothmm, feats, outdir, roisdir, roilabsf, fsmethod, otype):

    #create outdir if it does not exist
    if not outdir:
        outdir = os.path.join(datadir, 'oasis_' + feats + '_' + fsmethod + 'feats')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    #output base file name
    outbasename = 'oasis_' + feats + '_' + fsmethod
    if smoothmm > 0:
        outbasename += '_' + str(smoothmm) + 'mm'

    #fsmethod
    fsgrid = None
    #if fsmethod != 'stats' and fsmethod != 'none' and fsmethod != 'hist3d':
    #    fsmethod, fsgrid = get_fsmethod (fsmethod, n_feats, n_subjs)

    roilst = None
    if roilabsf:
        #get info from ROIs
        aalinfo = np.loadtxt (roilabsf, dtype=str)
        roilst  = aalinfo[:,0]
        n_rois  = len(roilst)

        #get a list of the aal roi volumes
        roiflst = dir_search('aal.smooth*', roisdir)
        roiflst = join_path_to_filelist (roisdir, roiflst)

        #create roilst mask
        msk = create_rois_mask (roilst, roiflst)

    #load all data
    data, y, scores, imgsiz, indices = load_data (subjlstf, datadir, msk, smoothmm)

    #For now I'm doing only two class classification:
    y = np.array(y)
    y[y > 0] = 1
    y = y.astype(int)

    #create space for all features and read from subjects
    if roilst:
        create_feature_sets (fsmethod, fsgrid, data, msk, y, roilst, roiflst, aalinfo, outdir, outbasename, otype)
    else:
        create_feature_sets (fsmethod, fsgrid, data, msk, y, bins, outdir, outbasename, otype)

#-------------------------------------------------------------------------------
def main():

    parser  = set_parser()

    try:
       args = parser.parse_args ()
    except argparse.ArgumentError, exc:
       print (exc.message + '\n' + exc.argument)
       parser.error(str(msg))
       return -1

    feats     = args.feats.strip()
    datadir   = args.datadir.strip()
    outdir    = args.outdir.strip()
    roisdir   = args.roisdir.strip()
    roilabsf  = args.roilabsf.strip()
    subjlstf  = args.subjlstf.strip()
    fsmethod  = args.fsmethod.strip()
    otype     = args.otype.strip()
    smoothmm  = args.smoothmm
    bins      = args.bins
    verbose   = args.verbosity

    #logging config
    au.setup_logger(verbose)

    return main_do (datadir, subjlstf, bins, smoothmm, feats, outdir, roisdir, roilabsf, fsmethod, otype)

#-------------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())


