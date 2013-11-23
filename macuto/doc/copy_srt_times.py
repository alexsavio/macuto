#!/usr/bin/python

import os
from chardet.universaldetector import UniversalDetector
from pysrt import SubRipFile

def get_file_encoding (filepath):
    detector = UniversalDetector()
    detector.reset()
    for line in file(filepath, 'rb'):
        detector.feed(line)
        if detector.done: break
    detector.close()

    return detector.result['encoding']

'''
wd    = '/home/alexandre/Desktop/'
srtf1 = wd + os.path.sep + 'Le.Nom.Des.Gens.2011.FRENCH.DVDRiP.XViD-FiCTiON.br.srt'

srtf2 = wd + os.path.sep + 'Le.Nom.Des.Gens.2011.FRENCH.DVDRiP.XViD-FiCTiON.spa.srt'

strf3 = wd + os.path.sep + 'Le.Nom.Des.Gens.2011.FRENCH.DVDRiP.XViD-FiCTiON.eng.srt'

srcf = srtf1

dstf = srtf2

of = srtf2 + '.settime'
'''

#read source file
srcenc  = get_file_encoding(srcf)
srcsubs = SubRipFile.open  (srcf, encoding=srcenc)

#read destination file
dstenc  = get_file_encoding(dstf)
dstsubs = SubRipFile.open  (dstf, encoding=dstenc)

'''
SubRipFile are list-like objects of SubRipItem instances:

>>> len(first_sub)
>>> first_sub = subs[0]

SubRipItem instances are editable just like pure Python objects:

>>> first_sub.text = "Hello World !"
>>> first_sub.start.seconds = 20
>>> first_sub.end.minutes = 5

Shifting:

>>> subs.shift(seconds=-2) # Move all subs 2 seconds earlier
>>> subs.shift(minutes=1)  # Move all subs 1 minutes later
>>> subs.shift(ratio=25/23.9) # convert a 23.9 fps subtitle in 25 fps
>>> first_sub.shift(seconds=1) # Move the first sub 1 second later
>>> first_sub.start += {'seconds': -1} # Make the first sub start 1 second earlier

Removing:

>>> del subs[12]

Slicing:

>>> part = subs.slice(starts_after={'minutes': 2, seconds': 30}, ends_before={'minutes': 3, 'seconds': 40})
>>> part.shift(seconds=-2)

Saving changes:

>>> subs.save('other/path.srt', encoding='utf-8')
'''


