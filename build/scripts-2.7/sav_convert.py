#!/home/ayerdi/envs/mypy2/bin/python

import os
import baker
from path import path
from macuto.storage import (sav_to_pandas_rpy2,
                            sav_to_pandas_savreader)

@baker.command(params={"inputfile": "Path to the .sav file to be transformed",
                       "outputfile": "Path to the output file",
                       "otype": """Output file type. Choices: 'csv', 'hdf',
                                                           'stata', 'json',
                                                           'pickle' 'excel', 
                                                           'html'. Default: 'csv'""",
                       "method": """Which conversion method to use. 
                                    Choices: 'rpy2' to use Pandas Rpy2 wrapper
                                          or 'savread' to use savReaderWriter"""},
               shortopts={'inputfile': 'i', 'outputfile': 'o', 
                          'method': 'm', 'otype': 't'})
def transform(inputfile, outputfile=None, method='rpy2', otype='csv'):
    """ Transforms the input .sav file into other format.
    If you don't specify an outputfile, it will use the
    inputfile and change its extension to .csv
    """
    assert(os.path.isfile(inputfile))
    assert(method=='rpy2' or method=='savread')

    if method == 'rpy2':
        df = sav_to_pandas_rpy2(inputfile)
    elif method == 'savread':
        df = sav_to_pandas_savreader(inputfile)

    if outputfile is None:
        outputfile = inputfile.replace(path(inputfile).ext, '')

    if otype == 'csv':
        oext = '.csv'
        df.to_csv(outputfile + oext)
    elif otype == 'hdf':
        oext = '.h5'
        df.to_hdf(outputfile + oext, os.path.basename(outputfile))
    elif otype == 'stata':
        oext = '.dta'
        df.to_stata(outputfile + oext)
    elif otype == 'json':
        oext = '.json'
        df.to_json(outputfile + oext)
    elif otype == 'pickle':
        oext = '.pickle'
        df.to_pickle(outputfile + oext)
    elif otype == 'excel':
        oext = '.xls'
        df.to_excel(outputfile + oext)
    elif otype == 'html':
        oext = '.html'
        df.to_html(outputfile + oext)
    else:
        oext = '.csv'
        df.to_csv(outputfile + oext)

if __name__ == '__main__':
    baker.run()

#    import os
#    wd = '/home/alexandre/Dropbox/Documents/projects/santiago'
#    f = 'datos_clinicos_140221.sav'

#    outf = f.replace('.sav', '.csv')
#    sav2csv(os.path.join(wd, f), os.path.join(wd, outf))
