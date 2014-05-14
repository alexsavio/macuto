#!/usr/bin/env python

datadir = '/media/alexandre/cobre/santiago/data'

regex = 'anat*'

import os
import nipype

# Comment the following section to increase verbosity of output
nipype.config.set('logging', 'workflow_level', 'CRITICAL')
nipype.config.set('logging', 'interface_level', 'CRITICAL')
nipype.logging.update_logging(nipype.config)

from nipype import SelectFiles, Node

templates = dict(anat=os.path.join(datadir, '{subject_id}', 'anat.nii.gz'))
dg = Node(SelectFiles(templates), "selectfiles")
dg.inputs.subject_id = "15817"
dg.inputs.run = [2, 4]

import nipype.interfaces.fsl as fsl
mybet = fsl.BET(in_file='foo.nii', out_file='bar.nii')
result = mybet.run()