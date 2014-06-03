# coding=utf-8
#-------------------------------------------------------------------------------

#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#
#2013, Alexandre Manhaes Savio
#Use this at your own risk!
#-------------------------------------------------------------------------------

import os
import sys
import logging

import numpy as np
import nibabel as nib
from sklearn.preprocessing import LabelEncoder

from ..files.names import parse_subjects_list
from ..files.names import grep_one

log = logging.getLogger(__name__)


def load_data(subjsf, datadir, maskf, labelsf=None):
    """

    @param subjsf:
    @param datadir:
    @param maskf:
    @param labelsf:
    @return:
    x, y, scores, imgsiz, msk, indices
    """

    #loading mask
    msk     = nib.load(maskf).get_data()
    n_vox   = np.sum  (msk > 0)
    indices = np.where(msk > 0)

    #reading subjects list
    [scores, subjs] = parse_subjects_list(subjsf, datadir, labelsf=labelsf)
    scores = np.array(scores)

    imgsiz  = nib.load(subjs[0]).shape
    dtype   = nib.load(subjs[0]).get_data_dtype()
    n_subjs = len(subjs)

    #checking mask and first subject dimensions match
    if imgsiz != msk.shape:
        msg = 'Subject image {0} and mask {1} dimensions do ' \
              'not coincide.'.format(subjs[0], maskf)
        log.error(msg)
        raise(ValueError(msg))

    #relabeling scores to integers, if needed
    if not np.all(scores.astype(int) == scores):
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

    y = y.astype(int)

    #loading data
    log.info('Loading data...')
    x = np.zeros((n_subjs, n_vox), dtype=dtype)
    for f in np.arange(n_subjs):
        imf = subjs[f]
        log.info('Reading ' + imf)

        img = nib.load(imf).get_data()
        x[f, :] = img[indices]

    return x, y, scores, imgsiz, msk, indices


def write_svmperf_dat(filename, dataname, data, labels):
    """ ARFFWRITE  Writes numeric data as an SVM Perf .dat formatted file.

    USAGE:
          writeSVMPerfDAT(fileName,dataName,data,labels);
      example: writeSVMPerfDAT('myDB.arff','db-name',mydata,labels);

    INPUT:
          filename:       String. Out file name.
          dataname:       String. A name for the database.
          data:           Numeric data matrix.
          labels:         Vector that indicates class, which must be {1,-1} and
                          length as rows of data.

    DETAILS:
          Writes data using 4 digits to the right of the decimal point.

    EXAMPLE:

          // Having face images in vector-rows in a matrix
          [m n]=size(images);
          attnames=[1:n]; // If attribute names are trivial, just ennumerate them
          images('ImagesForSVMPerf.dat','Face Images',images,labels);

    SVM Perf .dat syntax:
    <line> .=. <target> <feature>:<value> <feature>:<value> ... <feature>:<value> # <info>
    <target> .=. {+1,-1}
    <feature> .=. <integer>
    <value> .=. <float>
    <info> .=. <string>
    """

    nsamps = data.shape[0]
    nfeats = data.shape[1]
    nlabs  = len(labels)
    if nlabs != nsamps:
        err = 'Dimensions (rows) of data -1 must agree with number of labels!'
        log.error(err)
        raise IOError(err)

    label_chk = False
    if nlabs > 1:
        label_chk = np.sum((labels == 1) + (labels == -1))
        label_chk = label_chk != nsamps
    elif nlabs == 1:
        label_chk = labels[0] == 1 or labels[0] == -1

    if not label_chk:
        err = ':abels vector should have only -1 or 1 values!'
        log.error(err)
        raise IOError(err)

    # Open/create file
    with open(filename, 'w') as fd:

        # Write headings
        fd.write('#' + dataname + '\n')

        # Writing format for the data (comma delimited matrix)
        format = '%+d '
        for i in np.arange(nfeats):
            format += str(i+1) + ':%6.4f '

        labels = labels.reshape(nsamps, 1)
        data = np.concatenate((labels, data), axis=1)

        # Write data
        np.savetxt(fd, data, format)


def write_arff(filename, dataname, featnames, data, labels):
    """
    ARFFWRITE  Writes numeric data as an arff formatted file.

    USAGE:
          writeARFF(fileName,dataName,attNames,data);
      example: writeARFF('myDB.arff','db-name',myattnames,mydata);

    INPUT:
          filename:       String. Out File name.
          dataname:       String. A name for the database.
          featnames:      Array of numbers or cell of strings. Names of each
                          attribute.
          data:           Numeric data matrix.
          labels:         Vector that indicates class


    DETAILS:
          Writes data using 4 digits to the right of the decimal point.

    EXAMPLE:

          // Having face images in vector-rows in a matrix
          [m n]=size(images);
          attnames=[1:n]; // If attribute names are trivial, just enumerate them
          images('ImagesForWeka.arff','Face Images',attnames,images);

     ARFF (Attribute-Relation File Format) syntax:
     http://weka.wikispaces.com/ARFF+%28stable+version%29
     """

    # Check for input data
    nsamps = data.shape[0]
    nfeats = data.shape[1]
    if nfeats != len(featnames):
        err = 'Dimensions (column) of data must agree ' \
              'with number of variable name!'
        log.error(err)
        raise IOError(err)

    # Open/create file
    with open(filename, 'w') as fd:

        #Write headings
        fd.write('@RELATION ' + dataname + '\n')

        # Writing feature names in the arff file format.
        for i in featnames:
            fd.write('@ATTRIBUTE ' + str(i) + ' NUMERIC\n')

        # Write classes
        #classes = str(np.unique(labels)).replace('[','').replace(']','');
        classes = np.unique(labels).astype(int)
        classes = classes.reshape(1, len(classes))
        fd.write('@ATTRIBUTE class {')
        np.savetxt(fd, classes, fmt='%d', delimiter=',', newline='}')
        fd.write('\n')

        # Write data
        fd.write('@DATA\n')

        # Writing format for the data (comma delimited matrix)
        fmt = ''
        for i in featnames:
            fmt += ' %6.4f,'

        fmt += ' %d'

        labels = labels.reshape(nsamps, 1)
        data = np.concatenate((data, labels), axis=1)

        # Write data
        np.savetxt(fd, data, fmt)


def read_svmperf_results(logpath, predspath='', testlabels=''):
    """
    Returns ['Accuracy', 'Precision', 'Recall', 'F1', 'PRBEP', 'ROCArea', 'AvgPrec', 'Specificity', 'Brier score']

    @param logpath: string
    @param predspath: string
    @param testlabels: string
    @return:
    """

    if not os.path.exists(logpath):
        err = 'read_svmperf_results: Could not find file ' + logpath
        raise IOError(err)

    if predspath:
        if not os.path.exists(predspath):
            err = 'read_svmperf_results: Could not find file ' + predspath
            raise IOError(err)

    results = np.zeros(9, dtype=float)

    results[0] = float(grep_one('Accuracy',  logpath).strip().split(':')[1])
    results[1] = float(grep_one('Precision', logpath).strip().split(':')[1])
    results[2] = float(grep_one('Recall',    logpath).strip().split(':')[1])
    results[3] = float(grep_one('F1',        logpath).strip().split(':')[1])
    results[4] = float(grep_one('PRBEP',     logpath).strip().split(':')[1])
    results[5] = float(grep_one('ROCArea',   logpath).strip().split(':')[1])
    results[6] = float(grep_one('AvgPrec',   logpath).strip().split(':')[1])

    predsok = False
    if testlabels:
        try:
            preds   = np.loadtxt(predspath, dtype=float)
            predsok = True
        except IOError as err:
            log.error(err.msg())
            pass

    if predsok:
        res = np.sign(preds)
        n   = len(testlabels)

        if n == 1:
            lbs = testlabels[0]

        tp = 0
        fp = 0
        tn = 0
        fn = 0

        if   lbs ==  1 and res ==  1 : tp = 1
        elif lbs == -1 and res ==  1 : fp = 1
        elif lbs == -1 and res == -1 : tn = 1
        elif lbs ==  1 and res == -1 : fn = 1

        else:
            lbs = np.array(testlabels)
            tp = np.sum(lbs[lbs ==  1] == res[lbs ==  1])
            fp = np.sum(lbs[lbs == -1] != res[lbs == -1])
            tn = np.sum(lbs[lbs == -1] == res[lbs == -1])
            fn = np.sum(lbs[lbs ==  1] != res[lbs ==  1])

        accuracy = (float(tp+tn)/float(n)) * 100

        if ((tp+fp) > 0): precision   = (float(tp)/float(tp+fp)) * 100
        else            : precision   = results[1]

        if ((tp+fn) > 0): recall      = (float(tp)/float(tp+fn)) * 100
        else            : recall      = results[2]

        sensitivity = recall

        if ((tn+fp) > 0) : specificity = (float(tn)/float(tn+fp)) * 100
        else             : specificity = 0

        #Brier score
        if n == 1:
            if lbs == -1: lbs = 0
            if res == -1: res = 0
        else:
            lbs[lbs == -1] = 0
            res[res == -1] = 0

        brier_score = (1/float(n)) * np.sum(np.square(res - lbs))

        results[0] = accuracy
        results[1] = precision
        results[2] = recall
        results[7] = specificity
        results[8] = brier_score

    return results