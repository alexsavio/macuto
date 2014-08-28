import logging
from itertools import chain
from collections import OrderedDict, Callable

log = logging.getLogger(__name__)


def dictify(a_named_tuple):
    """Transform a named tuple into a dictionary"""
    return dict((s, getattr(a_named_tuple, s)) for s in a_named_tuple._fields)


def merge_dict_of_lists(adict, indices, pop_later=True, copy=True):
    """Extend the within a dict of lists. The indices will indicate which
    list have to be extended by which other list.

    Parameters
    ----------
    adict: OrderedDict
        An ordered dictionary of lists

    indices: list or tuple of 2 iterables of int, bot having the same length
        The indices of the lists that have to be merged, both iterables items
         will be read pair by pair, the first is the index to the list that
         will be extended with the list of the second index.
         The indices can be constructed with Numpy e.g.,
         indices = np.where(square_matrix)

    pop_later: bool
        If True will oop out the lists that are indicated in the second
         list of indices.

    copy: bool
        If True will perform a deep copy of the input adict before
         modifying it, hence not changing the original input.

    Returns
    -------
    Dictionary of lists

    Raises
    ------
    IndexError
        If the indices are out of range
    """
    def check_indices(idxs, x):
        for i in chain(*idxs):
            if i < 0 or i >= x:
                raise IndexError("Given indices are out of dict range.")

    try:
        check_indices(indices, len(adict))
    except IndexError as ie:
        log.exception('Error: Indices check did not pass')
        raise

    rdict = adict.copy() if copy else adict

    dict_keys = list(rdict.keys())
    for i, j in zip(*indices):
        rdict[dict_keys[i]].extend(rdict[dict_keys[j]])

    if pop_later:
        for i, j in zip(*indices):
            rdict.pop(dict_keys[j])

    return rdict


def append_dict_values(list_of_dicts, keys=None):
    """
    Return a dict of lists from a list of dicts with the same keys.
    For each dict in list_of_dicts with look for the values of the
    given keys and append it to the output dict.

    Parameters
    ----------
    list_of_dicts: list of dicts

    keys: list of str
        List of keys to create in the output dict
        If None will use all keys in the first element of list_of_dicts
    Returns
    -------
    DefaultOrderedDict of lists
    """
    if keys is None:
        try:
            keys = list(list_of_dicts[0].keys())
        except IndexError as ie:
            log.exception('Could not get the first element of the list.')
            raise

    dict_of_lists = DefaultOrderedDict(list)
    for d in list_of_dicts:
        for k in keys:
            try:
                dict_of_lists[k].append(d[k])
            except KeyError as ke:
                log.exception('Error looking for key {} in dict.'.format(k))
                raise
    return dict_of_lists


class ItemSet(object):

    def __iter__(self):
        return self.items.__iter__()

    def __next__(self):
        return self.items.__next__()

    def next(self):
        return self.items.next()

    def __getitem__(self, item):
        if hasattr(self.items, '__getitem__'):
            return self.items[item]
        else:
            msg = 'Item set has no __getitem__ implemented.'
            log.exception(msg)
            raise RuntimeError(msg)

    def __len__(self):
        return len(self.items)

    def save_to_file(self, file_path, var_name=['itemset']):
        from .storage import ExportData

        data_exporter = ExportData()
        data_exporter.save_varlist(file_path, var_name, [self])

    def extend(self, other_set):
        if isinstance(self.items, list):
            self.items.exted(other_set)
        elif isinstance(self.items, set):
            self.items.union(other_set)
        else:
            msg = 'ItemSet item has no extend implemented.'
            log.exception(msg)
            raise RuntimeError(msg)


class DefaultOrderedDict(OrderedDict):
    """An defauldict and OrderedDict.
    """
    def __init__(self, default_factory=None, *a, **kw):
        if (default_factory is not None and
                not isinstance(default_factory, Callable)):
            raise TypeError('first argument must be callable')
        OrderedDict.__init__(self, *a, **kw)
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        if self.default_factory is None:
            args = tuple()
        else:
            args = self.default_factory,
        return type(self), args, None, None, iter(self.items())

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)

    def __deepcopy__(self, memo):
        import copy
        return type(self)(self.default_factory,
                          copy.deepcopy(self.items()))

    def __repr__(self):
        return 'OrderedDefaultDict(%s, %s)' % (self.default_factory,
                                               OrderedDict.__repr__(self))
