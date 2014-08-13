import logging
from collections import OrderedDict, Callable

log = logging.getLogger(__name__)


def dictify(a_named_tuple):
    """Transforms a named tuple into a dictionary"""
    return dict((s, getattr(a_named_tuple, s)) for s in a_named_tuple._fields)


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
            raise log.exception('Item set has no __getitem__ implemented.')

    def __len__(self):
        return len(self.items)

    def save_to_file(self, out_path):
        import pickle
        with open(out_path, 'w') as f:
            for member in vars(self):
                pickle.dump(member, f)


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
