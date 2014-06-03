#!/home/ayerdi/envs/mypy2/bin/python

import os
import baker
import logging
from path import path

from macuto.scriptutils import whoami

#logging config
logging.basicConfig(level=logging.DEBUG, filename='filetree.log',
                    format="%(asctime)-15s %(message)s")
log = logging.getLogger('filetree')


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

    if not os.path.exists(destpath):
        log.info('Creating folder {0}'.format(destpath))
        path(destpath).makedirs_p()

    from macuto.files.file_tree_map import FileTreeMap
    file_map = FileTreeMap()
    file_map.from_config_file(configfile)

    if sub_node:
        sub_map = file_map.get_node(sub_node)
        if not sub_map:
            log.error('Could not find sub node {0}'.format(sub_node))
            return -1

        file_map._filetree = {}
        file_map._filetree[sub_node] = sub_map

    file_map.copy_to(destpath, overwrite=overwrite)


if __name__ == '__main__':
    baker.run()