#!/usr/bin/env python

import baker

@baker.command(default=True,
               shortopts={'configfile': 'c',
                          'destpath': 'o',})
def copy(configfile, destpath):
    """Copies the files in the built file tree map
    to despath.

    :param configfile: string
     Path to the FileTreeMap config file

    :param destpath: string
     Path to the files destination

    """
    from macuto.files.file_tree_map import FileTreeMap

    file_map = FileTreeMap()
    file_map.from_config_file(configfile)
    file_map.copy_to(destpath)

if __name__ == '__main__':
    baker.run()
