import os
import fnmatch
from functools import reduce
from collections import OrderedDict
from path import path

#except:
#    import shutil
#    from pathlib import Path as path
#    path.copyfile = shutil.copyfile

from ..exceptions import *
from ..strings import (match_list,
                       is_valid_regex)

from .names import get_extension

log = logging.getLogger(__name__)


class FileTreeMapError(LoggedError):
    pass


def import_pyfile(filepath, mod_name=None):
    """
    Imports the contents of filepath as a Python module.

    :param filepath: string

    :param mod_name: string
    Name of the module when imported

    :return: module
    Imported module
    """
    import sys
    if sys.version_info.major == 3:
        import importlib.machinery
        loader = importlib.machinery.SourceFileLoader('', filepath)
        mod = loader.load_module(mod_name)
    else:
        import imp
        mod = imp.load_source(mod_name, filepath)

    return mod


def is_regex(string):
    """
    TODO: improve this!

    Returns True if the given string is considered a regular expression,
    False otherwise.
    It will be considered a regex if starts with a non alphabetic character
    and then correctly compiled by re.compile

    :param string: str

    """
    is_regex = False
    regex_chars = ['\\', '(', '+', '^', '$']
    for c in regex_chars:
        if string.find(c) > -1:
            return is_valid_regex(string)
    return is_regex


def is_fnmatch_regex(string):
    """
    Returns True if the given string is considered a fnmatch 
    regular expression, False otherwise.
    It will look for 

    :param string: str

    """
    is_regex = False
    regex_chars = ['!', '*', '$']
    for c in regex_chars:
        if string.find(c) > -1:
            return True
    return is_regex


def filter_list(lst, pattern):
    """
    Filters the lst using pattern.
    If pattern starts with '(' it will be considered a re regular expression,
    otherwise it will use fnmatch filter.

    :param lst: list of strings

    :param pattern: string

    :return: list of strings
    Filtered list of strings
    """
    if is_fnmatch_regex(pattern) and not is_regex(pattern):
        #use fnmatch
        log.info('Using fnmatch for {0}'.format(pattern))
        filst = fnmatch.filter(lst, pattern)

    else:
        #use re
        log.info('Using regex match for {0}'.format(pattern))
        filst = match_list(lst, pattern)

    if filst:
        filst.sort()

    return filst


def remove_hidden_files(file_lst):
    '''
    Removes the filenames that start with '.'

    :param file_lst: list of strings

    :return: list of strings
    '''
    return [fnom for fnom in file_lst if not fnom.startswith('.')]


def get_subdict(adict, path, sep=os.sep):
    """
    Given a nested dictionary adict.
    This returns its childen just below the path.
    The path is a string composed of adict keys separated by sep.

    :param adict: nested dict

    :param path: str

    :param sep: str

    :return: dict or list or leaf of treemap

    """
    return reduce(adict.__class__.get, [p for p in path.split(sep) if p], adict)


def commonprefix(*args):
    return os.path.commonprefix(*args).rpartition(os.sep)[0]


def get_dict_leaves(data):
    """
    Given a nested dictionary, this returns all its leave elements in a list.

    :param adict:

    :return: list
    """
    result = []
    if isinstance(data, dict):
        for item in data.values():
            result.extend(get_dict_leaves(item))
    elif isinstance(data, list):
        result.extend(data)
    else:
        result.append(data)

    return result


def get_possible_paths(base_path, path_regex):
    """
    Looks for path_regex within base_path. Each match is append
    in the returned list.
    path_regex may contain subfolder structure.
    If any part of the folder structure is a 

    :param base_path: str

    :param path_regex: str

    :return list of strings
    """
    if not path_regex:
        return []

    if len(path_regex) < 1:
        return []

    if path_regex[0] == os.sep:
        path_regex = path_regex[1:]

    rest_files = ''
    if os.sep in path_regex:
        #split by os.sep
        node_names = path_regex.partition(os.sep)
        first_node = node_names[0]
        rest_nodes = node_names[2]

        folder_names = filter_list(os.listdir(base_path), first_node)

        for nom in folder_names:
            new_base = os.path.join(base_path, nom)
            if os.path.isdir(new_base):
                rest_files = get_possible_paths(new_base, rest_nodes)
    else:
        rest_files = filter_list(os.listdir(base_path), path_regex)

    files = []
    if rest_files:
        files = [os.path.join(base_path, f) for f in rest_files]

    return files


def process_tuple_node(basepath, treemap, ignore_hidden=True):
    """

    :param basepath:

    :param treemap: 2-tuple

    :param rootkey:

    :return:
    """
    if not isinstance(treemap, tuple):
        raise FileTreeMapError(log, 'treemap node must be a 2-tuple.')

    if len(treemap) != 2:
        raise FileTreeMapError(log, 'treemap node must be a 2-tuple.')

    file_nodes = OrderedDict()

    file_lst = os.listdir(basepath)

    if ignore_hidden:
        file_lst = remove_hidden_files(file_lst)

    children_names = filter_list(file_lst, treemap[0])
    child_map = treemap[1]

    if len(children_names) == 1:
        file_nodes.update(populate_subtree(os.path.join(basepath,
                                                        children_names[0]),
                                           child_map))
    else:
        for cname in children_names:
            child_basepath = os.path.join(basepath, cname)
            if os.path.isdir(child_basepath):
                subtrs = populate_subtree(child_basepath,
                                          child_map)
            else:
                subtrs = child_basepath

            if subtrs:
                file_nodes[cname] = subtrs

    return file_nodes


def populate_subtree(basepath, treemap, verbose=False):
    """

    :param path: str

    :param treemap: dict

    :return: dict
    """
    file_nodes = OrderedDict()

    if isinstance(treemap, tuple):
        try:
            file_nodes = process_tuple_node(basepath, treemap)
        except:
            raise FileTreeMapError('Error processing tuple node '
                                   '{0} {1}.'.format(basepath, treemap))

    if isinstance(treemap, list):
        for node in treemap:
            try:
                file_nodes.update(process_tuple_node(basepath, node))
            except:
                raise FileTreeMapError('Error processing tuple node '
                                       '{0} {1}.'.format(basepath, node))


    elif isinstance(treemap, dict):

        for k in treemap.keys():
            cname = k
            child_map = treemap[k]

            if isinstance(child_map, tuple) or isinstance(child_map, dict):
                try:
                    file_nodes[cname] = populate_subtree(basepath, child_map)
                except:
                    raise FileTreeMapError('Error populating subtree'
                                           '{0} {1}.'.format(basepath,
                                                             child_map))


            elif isinstance(child_map, str):
                if child_map[0] == os.sep:
                    raise FileTreeMapError('Error on node {0}. '
                                           'Relative paths should no start '
                                           'with "{1}"'.format(str(child_map),
                                                               os.sep))

                subpaths = get_possible_paths(basepath, child_map)
                if subpaths:
                    file_nodes[cname] = subpaths

    if verbose:
        log.info('{0} keys() -> {1}'.format(basepath, file_nodes.keys()))

    return file_nodes


class FileTreeMap(object):
    """
    This class is a registry of file lists.
    Works as a dictionary of file lists whose key is a given string, but
    giving further functionality related to relative paths, file indexing
    and searching.
    """
    def __init__(self):
        """
        :return:
        """
        self._filetree = {}
        self._treemap = {}
        self._basepath = ''
        self._ignore_regexes = []

    def __str__(self):
        """

        :return:
        """
        return 'root_path = {0}. \n filetree = {1}.'.format(self._basepath,
                                                            self._filetree)

    def from_config_file(self, config_file, verbose=False):
        """

        :param config_file: str
         Path to a configuration file.
         This file must declare a root_path and a filetree regex tree.

        :param verbose: bool
        """

        assert(os.path.isfile(config_file))
        self.__init__()
        self._basepath, self._treemap = self._import_config(config_file)
        self.update(verbose)

    def from_dict(self, root_path, filetree, verbose=False):
        """

        :param root_path: string
        :param filetree: dict
        """
        self.__init__()
        self._basepath = root_path
        self._treemap = filetree
        self.update(verbose)

    def update(self, verbose=False):
        """
        """
        try:
            self._check_basic_config()
            self._filetree = populate_subtree(self._basepath,
                                              self._treemap,
                                              verbose)

            log.info('FileTreeMap created: \n {0}.'.format(str(self)))
        except Exception as e:
            raise

    def _check_basic_config(self):
        """
        """
        root_path = self._basepath
        if root_path:
            if not os.path.isabs(root_path) or not os.path.exists(root_path):
                raise FolderNotFound('The root path {0} does not '
                                     'exist.'.format(root_path))

    @staticmethod
    def create_folder(dirpath, overwrite=False):
        if not overwrite:
            while os.path.exists(dirpath):
                dirpath += '+'

        path(dirpath).mkdir_p()
        return dirpath

    @staticmethod
    def _import_config(filepath):
        """
        Imports filetree and root_path variable values from the filepath.

        :param filepath:
        :return: root_path and filetree
        """
        if not os.path.isfile(filepath):
            raise IOError('Data config file not found. '
                          'Got: {0}'.format(filepath))

        cfg = import_pyfile(filepath)

        if not hasattr(cfg, 'root_path'):
            raise KeyError('Config file root_path key not found.')

        if not hasattr(cfg, 'filetree'):
            raise KeyError('Config file filetree key not found.')

        return cfg.root_path, cfg.filetree

    def get_root_nodes(self):
        """
        Return a list of the names of the root nodes.
        """
        return self._filetree.keys()

    def get_node(self, nodepath=None):
        if nodepath is None:
            return self.get_root_nodes()
        return get_subdict(self._filetree, nodepath)

    def get_node_filepaths(self, nodepath):
        """
        Returns all leaves in filetree.
        """
        files = self.get_node(nodepath)
        return get_dict_leaves(files)

    def get_common_filepath(self, nodepath):
        """
        Returns the common filepath between all leaves in the filetree.
        """
        return commonprefix(self.get_node_filepaths(nodepath))

    def remove_nodes(self, pattern, adict):
        """
        Remove the nodes that match the pattern.
        """
        mydict = self._filetree if adict is None else adict

        if isinstance(mydict, dict):
            for nom in mydict.keys():
                if isinstance(mydict[nom], dict):
                    matchs = filter_list(mydict[nom], pattern)
                    for nom in matchs:
                        mydict = self.remove_nodes(pattern, mydict[nom])
                        mydict.pop(nom)
                else:
                    mydict[nom] = filter_list(mydict[nom], pattern)
        else:
            matchs = set(filter_list(mydict, pattern))
            mydict = set(mydict) - matchs

        return mydict

    def count_node_match(self, pattern, adict=None):
        """
        Return the number of nodes that match the pattern.

        :param pattern:

        :param adict:
        :return: int
        """
        mydict = self._filetree if adict is None else adict

        k = 0
        if isinstance(mydict, dict):
            names = mydict.keys()
            k += len(filter_list(names, pattern))
            for nom in names:
                k += self.count_node_match(pattern, mydict[nom])
        else:
            k = len(filter_list(mydict, pattern))

        return k

    @staticmethod
    def _transfer_files(adict, dirpath, rename_files=True,
                        one_file_folders=False, overwrite=False,
                        copy_method=path.copyfile, verbose_check=False):
        """
        """
        enabled = True
        if verbose_check:
            enabled = False

        for k in adict.keys():
            knode = adict[k]
            if isinstance(knode, dict):
                dest = path(dirpath).joinpath(k)

                if enabled:
                    dest = FileTreeMap.create_folder(dest, overwrite)

                log.info('Created folder {0}'.format(dest))
                FileTreeMap._transfer_files(knode, dest, rename_files,
                                            one_file_folders, overwrite,
                                            copy_method)

            elif isinstance(knode, list):
                if len(knode) == 0:
                    continue

                if len(knode) == 1:
                    src = path(knode[0])

                    if one_file_folders:
                        destdir = path(dirpath).joinpath(k)
                        if enabled:
                            destdir = FileTreeMap.create_folder(destdir,
                                                                overwrite)

                        log.info('Created one folder {0}'.format(destdir))
                    else:
                        destdir = path(dirpath)

                    if rename_files:
                        destf = k + get_extension(src)
                    else:
                        destf = src.basename()

                    destf = destdir.joinpath(destf)
                    if enabled:
                        copy_method(src, destf)
                    log.info('Copying file {0} to {1}'.format(src, destf))
                else:
                    destdir = path(dirpath).joinpath(k)
                    if enabled:
                        destdir = FileTreeMap.create_folder(destdir, overwrite)
                    log.info('Created one folder {0}'.format(destdir))

                    no = 1
                    for src in knode:
                        src = path(src)
                        if rename_files:
                            destf = str(no).zfill(5) + get_extension(src)
                        else:
                            destf = src.basename()

                        destf = destdir.joinpath(destf)
                        if enabled:
                            copy_method(src, destf)
                        log.info('Copying file {0} to {1}'.format(src, destf))
                        no += 1

    def copy_to(self, dirpath, rename_files=True,
                one_file_folders=False, overwrite=False, only_verbose=False):
        """
        """
        if not os.path.exists(dirpath):
            dirpath = self.create_folder(dirpath)

        self._transfer_files(self._filetree, dirpath,
                             rename_files, one_file_folders, overwrite,
                             verbose_check=only_verbose)

    def __iter__(self):
        return self._filetree.__iter__()

    #TODO correct these two next calls
    def __next__(self):
        return self._filetree.__next__()

    def next(self):
        return self._filetree.next()
