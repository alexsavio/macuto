#!/home/ayerdi/envs/mypy2/bin/python
from __future__ import unicode_literals

import os
import baker
from path import path
from macuto.storage import (sav_to_pandas_rpy2,
                            sav_to_pandas_savreader)

from macuto.files.names import add_extension_if_needed

@baker.command(name='sav',
               params={"inputfile": "Path to the .sav file to be transformed",
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
def convert_sav(inputfile, outputfile=None, method='rpy2', otype='csv'):
    """ Transforms the input .sav SPSS file into other format.
    If you don't specify an outputfile, it will use the
    inputfile and change its extension to .csv
    """
    assert(os.path.isfile(inputfile))
    assert(method=='rpy2' or method=='savread')

    if method == 'rpy2':
        df = sav_to_pandas_rpy2(inputfile)
    elif method == 'savread':
        df = sav_to_pandas_savreader(inputfile)

    otype_exts = {'csv': '.csv', 
                  'hdf': '.h5', 
                  'stata': '.dta',
                  'json': '.json',
                  'pickle': '.pickle',
                  'excel': '.xls',
                  'html': '.html'}

    if outputfile is None:
        outputfile = inputfile.replace(path(inputfile).ext, '')

    outputfile = add_extension_if_needed(outputfile, otype_exts[otype])

    if otype == 'csv':
        df.to_csv(outputfile)
    elif otype == 'hdf':
        df.to_hdf(outputfile, os.path.basename(outputfile))
    elif otype == 'stata':
        df.to_stata(outputfile)
    elif otype == 'json':
        df.to_json(outputfile)
    elif otype == 'pickle':
        df.to_pickle(outputfile)
    elif otype == 'excel':
        df.to_excel(outputfile)
    elif otype == 'html':
        df.to_html(outputfile)
    else:
        df.to_csv(outputfile)

if __name__ == '__main__':
    baker.run()

#    import os
#    wd = '/home/alexandre/Dropbox/Documents/projects/santiago'
#    f = 'datos_clinicos_140221.sav'

#    outf = f.replace('.sav', '.csv')
#    sav2csv(os.path.join(wd, f), os.path.join(wd, outf))
