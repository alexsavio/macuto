#!/usr/bin/env python

import os
import baker
from path import path

from macuto.scriptutils import whoami
from macuto.exceptions import *
from macuto.files.file_tree_map import FileTreeMapError

#logging config
log = logging.getLogger(__name__)


@baker.command(default=True,
               shortopts={'configfile': 'c',
                          'destpath': 'd',
                          'sub_node': 's',
                          'overwrite': 'o'})
def copy(configfile='', destpath='', overwrite=False, sub_node=''):
    """Copies the files in the built file tree map
    to despath.

    :param configfile: string
     Path to the FileTreeMap config file

    :param destpath: string
     Path to the files destination

    :param overwrite: bool
     Overwrite files if they already exist.

    :param sub_node: string
     Tree map configuration sub path.
     Will copy only the contents within this sub-node

    """
    log.info('Running {0} {1} {2}'.format(os.path.basename(__file__),
                                          whoami(),
                                          locals()))

    assert(os.path.isfile(configfile))

    if os.path.exists(destpath):
        if os.listdir(destpath):
            raise FolderAlreadyExists('Folder {0} already exists. Please clean '
                                      'it or change destpath.'.format(destpath))
    else:
        log.info('Creating folder {0}'.format(destpath))
        path(destpath).makedirs_p()

    from macuto.files.file_tree_map import FileTreeMap
    file_map = FileTreeMap()

    try:
        file_map.from_config_file(configfile)
    except Exception as e:
        raise FileTreeMapError(str(e))

    if sub_node:
        sub_map = file_map.get_node(sub_node)
        if not sub_map:
            raise FileTreeMapError('Could not find sub node '
                                   '{0}'.format(sub_node))

        file_map._filetree = {}
        file_map._filetree[sub_node] = sub_map

    try:
        file_map.copy_to(destpath, overwrite=overwrite)
    except Exception as e:
        raise FileTreeMapError(str(e))


if __name__ == '__main__':
    baker.run()