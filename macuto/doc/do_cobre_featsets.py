import os
import re
import sys
import socket
import commands
import argparse
import numpy as np
import nibabel as nib
import pickle
import commands

import scipy.stats as stats
import scipy.io as sio

import socket
hn = socket.gethostname()

sys.path.append('/home/alexandre/Dropbox/Documents/phd/work/aizkolari')
import aizkolari_utils as au

from IPython.core.debugger import Tracer; debug_here = Tracer()

#-------------------------------------------------------------------------------
def get_filepaths (temporal_filtering=True, global_nuis_correction=False):

    hn = socket.gethostname()

    if hn == 'finn':
        xd  = '/home/alexandre/Dropbox/Documents/phd/work/cobre'  #execution dir
        wd  = '/media/alexandre/iba/data/cobre'                   #working and output dir
        dd  = '/home/alexandre/Desktop/cobre'                     #data dir
        td  = '/usr/share/fsl/data/standard'                      #templates dir
    elif hn == 'buccaneer':
        xd  = '/home/alexandre/Dropbox/Documents/phd/work/cobre'
        #wd  = '/media/alexandre/iba/data/cobre'
        wd  = '/media/alexandre/bckp/data/cobre'
        dd  = os.path.join(wd, 'cpac/out/sym_links')
        td  = '/usr/share/fsl/data/standard'
    elif hn == 'corsair':
        xd  = '/home/alexandre/Dropbox/Documents/phd/work/cobre'
        wd  = '/home/alexandre/Data/cobre'
        dd  = os.path.join(wd, 'cpac/out/sym_links')
        td  = '/usr/share/fsl/data/standard'


    pipe1 = '_compcor_ncomponents_5_linear1.global1.motion1.compcor1.CSF_0.98_GM_0.7_WM_0.98'
    pipe2 = '_compcor_ncomponents_5_linear1.motion1.compcor1.CSF_0.98_GM_0.7_WM_0.98'

    #these ones failed subjects: 0040117, 0040127, 0040145
    pipe3 = '_compcor_ncomponents_5_pc11.linear1.wm1.motion1.gm1.compcor1.csf1_CSF_0.98_GM_0.7_WM_0.98'
    pipe4 = 'linear1.wm1.motion1.gm1.csf1_CSF_0.98_GM_0.7_WM_0.98'

    if temporal_filtering:     pipe = 'pipeline_HackettCity'
    else:                      pipe = 'pipeline_LitchfieldCity'

    if global_nuis_correction: pipe = os.path.join(pipe, pipe1)
    else:                      pipe = os.path.join(pipe, pipe2)

    dd = os.path.join(dd, pipe)

    #diagnostic data
    labelsf = os.path.join(xd, 'subj_diagnosis_labels.txt')

    #reference list file: phenotypic data
    phenof = os.path.join(xd, 'COBRE_phenotypic_data.csv')

    if not os.path.exists(os.path.join(wd, 'features')):
        os.mkdir(os.path.join(wd, 'features'))

    pipef = pipe.replace('/', '.')

    dataf = [
    os.path.join(wd, 'features/cobre_alff_Z_to_standard_smooth_' + pipef + '.npy'),
    os.path.join(wd, 'features/cobre_falff_Z_to_standard_smooth_' + pipef + '.npy'),
    os.path.join(wd, 'features/cobre_reho_Z_to_standard_smooth_' + pipef + '.npy'),
    os.path.join(wd, 'features/cobre_vmhc_z_score_' + pipef + '.npy'),
    os.path.join(wd, 'features/cobre_vmhc_z_score_stat_map_' + pipef + '.npy')
    ]

    masks = [
    os.path.join(td, 'MNI152_T1_3mm_brain_mask.nii.gz'),
    os.path.join(td, 'MNI152_T1_3mm_brain_mask.nii.gz'),
    os.path.join(td, 'MNI152_T1_3mm_brain_mask.nii.gz'),
    os.path.join(td, 'MNI152_T1_2mm_brain_mask.nii.gz'),
    os.path.join(td, 'MNI152_T1_2mm_brain_mask.nii.gz'),
    ]

    dilmasks = [
    os.path.join(td, 'MNI152_T1_3mm_brain_mask_dil.nii.gz'),
    os.path.join(td, 'MNI152_T1_3mm_brain_mask_dil.nii.gz'),
    os.path.join(td, 'MNI152_T1_3mm_brain_mask_dil.nii.gz'),
    os.path.join(td, 'MNI152_T1_2mm_brain_mask_dil.nii.gz'),
    os.path.join(td, 'MNI152_T1_2mm_brain_mask_dil.nii.gz'),
    ]

    templates = [
    os.path.join(td, 'MNI152_T1_3mm_brain.nii.gz'),
    os.path.join(td, 'MNI152_T1_3mm_brain.nii.gz'),
    os.path.join(td, 'MNI152_T1_3mm_brain.nii.gz'),
    os.path.join(td, 'MNI152_T1_2mm_brain.nii.gz'),
    os.path.join(td, 'MNI152_T1_2mm_brain.nii.gz'),
    ]

    return wd, xd, dd, labelsf, phenof, dataf, masks, dilmasks, templates, pipe

#-------------------------------------------------------------------------------
def get_feats (subj_ids, flist, indices):
    '''
    choose files from flist in the same order as in subj_ids
    and extract from each file the values within indices.
    Parameters
    ===========
    subj_ids: list of strings.
        List with identification of the subjects.
    flist: list of files
        List of files to be opened, subject id should be in these file paths.
    indices: Numpy array 3xM
        Array with the indices of the mask to extract the data from the files.
    '''
    n_subjs = len(flist)
    n_vox   = len(indices[0])
    feats   = np.zeros((n_subjs,n_vox), dtype=nib.load(flist[0]).get_data_dtype())

    for i in range(len(subj_ids)):
        sf = find_subject(flist, subj_ids[i])
        if sf:
            #print (str(i) + ' -> ' + subj_ids[i] )
            vol = nib.load(sf).get_data()
            feats[i,:] = vol[indices]
        else:
            i -= 1

    return feats


#-------------------------------------------------------------------------------
def load_mask (maskf):
    '''
    Loads a Nifti mask volume
    RETURNS
    The volume array and indices where mask > 0
    '''
    mask    = nib.load(maskf).get_data()
    indices = np.where(mask > 0)
    n_vox   = np.sum(mask > 0)
    return mask, indices

#-------------------------------------------------------------------------------
def find_subject (flist, subj):
    for i in flist:
        #if subj in i.split('/'):
        if i.rfind(subj) > 0:
            return i
    return None

#-------------------------------------------------------------------------------
def find_subjects (flist, subj_ids):
    files = []
    for s in subj_ids:
        f = find_subject(flist, s.split('_')[0])
        if not f:
            print "Could not find file for: " + s
        else:
            files.append(f)
    return files

#-------------------------------------------------------------------------------
def find_name_sh (regex, wd='.', args=None):
    comm = 'find ' + wd + ' -name ' + regex
    if args:
        comm += ' ' + args

    out = commands.getoutput(comm)
    lst = out.split()

    return lst

#-------------------------------------------------------------------------------
def save_nibabel (ofname, vol, affine, header=None):
    '''Saves a volume into a Nifti (.nii.gz) file.

    Parameters
    ===========
    ofname: string
        File relative path and name
    vol: Numpy 3D or 4D array
        Volume with the data to be saved.
    affine: 4x4 Numpy array
        Array with the affine transform of the file.
    header: nibabel.nifti1.Nifti1Header, optional
        Header for the file, optional but recommended.
    '''
    log.debug('Saving nifti file: ' + ofname)
    ni = nib.Nifti1Image(vol, affine, header)
    nib.save(ni, ofname)

#-------------------------------------------------------------------------------
#load phenotypic data
def extract_pheno_info (pheno_file):
    phenos = np.loadtxt(pheno_file, dtype=str, comments='#')
    pheno  = []
    for i in np.arange(len(phenos)):
        pheno.append(phenos[i].split(','))
    del phenos

    return np.array(pheno)

#-------------------------------------------------------------------------------
def do_slicesdir (file_list, mask_file, dest_dir, show_browser=True):

    listf = 'subjlist.txt'

    oldd  = os.path.abspath(os.curdir)
    newd  = os.path.dirname(os.path.commonprefix(file_list)) + '/'
    flist = [s.replace(newd,'') for s in file_list]

    if not os.path.exists(newd):
        os.mkdir(newd)

    os.chdir(newd)
    np.savetxt(listf, flist, fmt='%s')

    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    commands.getoutput('slicesdir -p ' + mask_file + ' `cat '+ listf + '`')
    commands.getoutput('cp slicesdir/* ' + dest_dir)
    commands.getoutput('rm -r slicesdir')

    os.remove(listf)
    os.chdir(oldd)

    if show_browser:
        if au.which ('chromium-browser'):
            browser = 'chromium-browser'
        elif au.which ('firefox'):
            browser = 'firefox'
        elif au.which ('chrome'):
            browser = 'chrome'

        commands.getoutput(browser + ' file:///' + dest_dir + '/index.html')

#-------------------------------------------------------------------------------
'''
import sys
sys.path.append('/home/alexandre/Dropbox/Documents/phd/work/cobre')

import do_cobre_featsets as cs
from sklearn.grid_search import ParameterGrid

slicesdir    = True
show_browser = True

temporal_filtering     = [True, False]
global_nuis_correction = [True, False]
preprocess_grid = {'temporal_filtering': [True, False], 'global_nuis_correction': [True, False]}

for j in list(ParameterGrid(preprocess_grid)):
    cs.do_featset (**j, slicesdir, show_browser)
'''
def do_featset (temporal_filtering=True, global_nuis_correction=False, slicesdir=False, show_browser=False):

    # Files of interest (FOIs)
    wd, xd, dd, labelsf, phenof, dataf, masks, dilmasks, templates, pipe = get_filepaths(temporal_filtering, global_nuis_correction)

    #functional masks
    fmask3f    = masks[0]
    fmask2f    = masks[3]

    #anatomical mask
    amaskf    = os.path.join(wd, 'MNI152_T1_1mm_brain_mask.nii.gz')

    #regexes
    #functional
    rehox = 'reho_Z_to_standard_smooth.nii.gz'

    vmhcx  = 'vmhc_z_score.nii.gz'
    vmhcsx = 'vmhc_z_score_stat_map.nii.gz'

    alffx  =  'alff_Z_to_standard_smooth.nii.gz'
    falffx = 'falff_Z_to_standard_smooth.nii.gz'

    funcfiltx = 'functional_freq_filtered.nii.gz'
    funcmnix  = 'functional_mni.nii.gz'

    #anatomical
    gmx = 'segment_pve_1.nii.gz'

    #anatomical registration
    mnitoanat_linx    = 'anatomical_to_mni_linear_xfm.mat'
    mnitoanat_nonlinx = 'anatomical_to_mni_nonlinear_xfm.nii.gz'

    anattomni_linx    = 'mni_to_anatomical_linear_xfm.mat'
    anattomni_nonlinx = 'mni_to_anatomical_nonlinear_xfm.nii.gz'

    #what do you want?
    fregex = [rehox, vmhcx, vmhcsx, alffx, falffx]

    #load phenotypic data
    pheno = extract_pheno_info (phenof)

    #which mask? anatomical or functional?
    mask2f   = fmask2f
    mask3f   = fmask3f

    #maskf   = amaskf
    mask2, indices2 = load_mask (mask2f)
    mask3, indices3 = load_mask (mask3f)

    for f in fregex:
        print f

        #find files
        ftypename = au.remove_ext(f)
        flist = find_name_sh(f, dd)
        flist.sort()

        #check if mask and subject sizes match
        shape = nib.load(flist[0]).shape
        if   mask2.shape == shape:
            mask, indices = mask2, indices2
            maskf = mask2f
        elif mask3.shape == shape:
            mask, indices = mask3, indices3
            maskf = mask3f
        else:
            print "Mask and subject sizes don't coincide! Exiting."
            exit()

        if slicesdir:
            sli_od = os.path.join(wd, 'slicesdir', 'cobre_' + ftypename + '.' + pipe.replace('/','.') + '_slicesdir')
            print ('Saving slicesdir in ' + sli_od)
            do_slicesdir (flist, maskf, sli_od, show_browser)

        #load and save files
        feats = get_feats (pheno[:,0], flist, indices)

        #save file
        of = os.path.join(wd, 'cobre_' + ftypename + '_' + pipe.replace('/','.'))
        print ('Saving ' + of + '.npy')
        np.save(of + '.npy', feats)


#-------------------------------------------------------------------------------
# USED FOR GRAPH FEATURES

def register_aal_to_functional (subj_dir, aal_path):

    sd = subj_dir

    anat_to_mni_nl     = os.path.join(sd, 'anatomical_to_mni_nonlinear_xfm/mprage_RPI_fieldwarp.nii.gz')
    anat_to_mni_nl_inv = os.path.join(sd, 'anatomical_to_mni_nonlinear_xfm/mprage_RPI_fieldwarp_inv.nii.gz')
    anat_brain         = os.path.join(sd, 'anatomical_brain/mprage_RPI_3dc.nii.gz')

    if not os.path.exists(anat_to_mni_nl_inv):
        print('Creating ' + anat_to_mni_nl_inv)
        commands.getoutput('invwarp -w ' + anat_to_mni_nl + ' -o ' + anat_to_mni_nl_inv + ' -r ' + anat_brain)

    aal_anatwarped   = os.path.join(sd, 'aal_3mm_anatomical.nii.gz')
    aal_funcwarped   = os.path.join(sd, 'aal_3mm_functional.nii.gz')

    anat_to_mni_mat  = os.path.join(sd, 'anatomical_to_mni_linear_xfm/mprage_RPI_3dc_flirt.mat')
    anat_to_func_mat = os.path.join(sd, 'anatomical_to_functional_xfm/_scan_rest_1_rest/mprage_RPI_3dc_flirt.mat')
    mean_func        = os.path.join(sd, 'mean_functional/_scan_rest_1_rest/rest_3dc_tshift_RPI_3dv_3dc_3dT.nii.gz')

    if not os.path.exists(aal_anatwarped):
        print('Creating ' + aal_anatwarped)
        commands.getoutput('applywarp --in=' + aal_path + ' --ref=' + anat_brain + ' --out=' + aal_anatwarped + ' --warp=' + anat_to_mni_nl_inv + ' --interp=nn')

    if not os.path.exists(aal_funcwarped):
        print('Creating ' + aal_funcwarped)
        getoutput.commands('applywarp --in=' + aal_anatwarped + ' --ref=' + mean_func + ' --out=' + aal_funcwarped + ' --premat=' + anat_to_func_mat + ' --interp=nn')

    return aal_funcwarped, aal_anatwarped

#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
'''
import sys
sys.path.append('/home/alexandre/Dropbox/Documents/phd/work/cobre')

import do_cobre_featsets as cs
from sklearn.grid_search import ParameterGrid

temporal_filtering     = [True, False]
global_nuis_correction = [True, False]
preprocess_grid = {'temporal_filtering': [True, False], 'global_nuis_correction': [True, False]}

#tssel_method  : 'mean', 'eigen', 'ilsia', 'cca'
#simil_measure : 'crosscorrelation', 'correlation', 'coherence', 'mean_coherence', 'mean_correlation'

#MY EXPERIMENT
#experiments_grid = {}
#experiments_grid.update({'normalize'    : ['percent', 'zscore']})
#experiments_grid.update({'average'      : [False]})
#experiments_grid.update({'tssel_method' : ['eigen']})
#experiments_grid.update({'simil_measure': ['correlation', 'coherence']})
#experiments_grid.update({'n_comps'      : [1]})
#experiments_grid.update({'filter'       : [None, {'lb': 0.001, 'ub': 0.01}]})

#FEKETE EXPERIMENT
experiments_grid = {}
experiments_grid.update({'normalize'    : ['percent', 'zscore']})
experiments_grid.update({'average'      : [False]})
experiments_grid.update({'tssel_method' : ['eigen_and_filtered', 'mean_and_filtered']})
experiments_grid.update({'simil_measure': ['correlation', 'coherence']})
experiments_grid.update({'n_comps'      : [1]})
experiments_grid.update({'filter'       : [None]})
experiments_grid.update({'fekete-wilf'  : [True]})

output_basename = 'connectivity_matrices'

for j in list(ParameterGrid(preprocess_grid)):
    outf = output_basename

    tf = j['temporal_filtering']
    gc = j['global_nuis_correction']
    outf += '.tmp_filt'    if tf else '.no_tmp_filt'
    outf += '.glob_correct' if gc else '.no_glob_correct'

    for params in list(ParameterGrid(experiments_grid)):
        print params
        cs.do_cobre_graph_featset (outf, tf, gc, **params)

'''
def do_cobre_graph_featset (output_basename, temporal_filtering, global_nuis_correction, **kwargs):
    import sys
    sys.path.append('/home/alexandre/Dropbox/Documents/phd/work/conn_matrices')
    import rsfmri_conn_matrices as conmat

    # Files of interest (FOIs)
    wd, xd, dd, labelsf, phenof, dataf, masks, dilmasks, templates, pipe = get_filepaths(temporal_filtering, global_nuis_correction)

    dd = os.path.join(wd, 'cpac/out/', os.path.dirname(pipe))

    #set AAL file path
    aal = os.path.join(xd, 'aal_3mm.nii.gz')

    funcs    = []
    aal_rois = []

    #output file path
    argstr   = '.'.join("{!s}={!r}".format(key,val).replace(' ','').replace('=', '_').replace("'",'') for (key,val) in kwargs.items())
    outfile  = output_basename + '.' + argstr
    outfile  = os.path.join(wd, outfile)
    outfile += '.pyshelf'

    if os.path.exists(outfile):
        print(outfile + ' already created.')
        return True

    #connectivity folder
    cd = os.path.join(wd, 'connectivity')

    if os.path.exists(cd):
        subjs = os.listdir(cd)
        subjs.sort()

        for s in subjs:
            sd  = os.path.join(cd, s)
            fnc = os.path.join(sd, 'rest_3dc_tshift_RPI_3dv_3dc_maths.nii.gz')
            aal_funcwarped = os.path.join(sd, 'aal_3mm_functional.nii.gz')

            funcs.append   (fnc)
            aal_rois.append(aal_funcwarped)

    else:
        #prepare data (invert deformation fields and warping)
        subjs = os.listdir(dd)
        subjs.sort()

        for s in subjs:
            sd = os.path.join(dd, s)

            #preprocess (register AAL to subject functional)
            aal_funcwarped, aal_anatwarped = register_aal_to_functional (sd, aal)

            #load processed data
            fnc = os.path.join(sd, 'preprocessed/_scan_rest_1_rest/rest_3dc_tshift_RPI_3dv_3dc_maths.nii.gz')

            funcs.append   (fnc)
            aal_rois.append(aal_funcwarped)

    #format dic to string
    print('Creating ' + outfile + '\n')
    conmat.save_connectivity_matrices(funcs, aal_rois, outfile, TR=None, **kwargs)


