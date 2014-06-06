# coding=utf-8
#-------------------------------------------------------------------------------

#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#
#2013, Alexandre Manhaes Savio
#Use this at your own risk!
#-------------------------------------------------------------------------------

import re
import numpy as np

def filter_objlist(olist, fieldname, fieldval):
    """
    Returns a list with of the objetcts in olist that have a fieldname valued as fieldval

    @param olist: list of objects
    @param fieldname: string
    @param fieldval: anything

    @return: list of objets
    """
    return [x for x in olist if getattr(x, fieldname) == fieldval]


def pretty_mapping(mapping, getterfunc=None):
    """
    Make pretty string from mapping

    Adjusts text column to print values on basis of longest key.
    Probably only sensible if keys are mainly strings.

    You can pass in a callable that does clever things to get the values
    out of the mapping, given the names.  By default, we just use
    ``__getitem__``

    This function has been copied from NiBabel:
    http://nipy.org/nibabel/
    Which has a MIT License

    Parameters
    ----------
    :param mapping : mapping
       implementing iterator returning keys and .items()
    :param getterfunc : None or callable
       callable taking two arguments, ``obj`` and ``key`` where ``obj``
       is the passed mapping.  If None, just use ``lambda obj, key:
       obj[key]``

    Returns
    -------
    :return str : string

    Examples
    --------
    >>> d = {'a key': 'a value'}
    >>> print(pretty_mapping(d))
    a key  : a value
    >>> class C(object): # to control ordering, show get_ method
    ...     def __iter__(self):
    ...         return iter(('short_field','longer_field'))
    ...     def __getitem__(self, key):
    ...         if key == 'short_field':
    ...             return 0
    ...         if key == 'longer_field':
    ...             return 'str'
    ...     def get_longer_field(self):
    ...         return 'method string'
    >>> def getter(obj, key):
    ...     # Look for any 'get_<name>' methods
    ...     try:
    ...         return obj.__getattribute__('get_' + key)()
    ...     except AttributeError:
    ...         return obj[key]
    >>> print(pretty_mapping(C(), getter))
    short_field   : 0
    longer_field  : method string
    """
    import numpy as np

    if getterfunc is None:
        getterfunc = lambda obj, key: obj[key]
    lens = [len(str(name)) for name in mapping]
    mxlen = np.max(lens)
    fmt = '%%-%ds  : %%s' % mxlen
    out = [fmt % (name, getterfunc(mapping, name)) for name in mapping]
    #for name in mapping:
    #    value = getterfunc(mapping, name)
    #    out.append(fmt % (name, value))
    return '\n'.join(out)


def filter_list(lst, filt):
    """
    :param lst: list
    :param filter: function
    Unary string filter function
    :return: list
    List of strings that passed the filter

    :example
    l = ['12123123', 'N123213']
    filt = re.compile('\d*').match
    nu_l = list_filter(l, filt)
    """
    return [m for s in lst for m in (filt(s),) if m]


def match_list(lst, pattern, group_names=[]):
    """
    @param lst: list of strings

    @param regex: string

    @param group_names: list of strings
    See re.MatchObject group docstring

    @return: list of strings
    Filtered list, with the strings that match the pattern
    """
    filtfn = re.compile(pattern).match
    filtlst = filter_list(lst, filtfn)
    if group_names is None:
        return [m.string for m in filtlst]
    else:
        return [m.group(group_names) for m in filtlst]


def search_list(lst, pattern):
    """
    @param pattern: string
    @param lst: list of strings
    @return: list of strings
    Filtered lists with the strings in which the pattern is found.

    """
    filt = re.compile(pattern).search
    return filter_list(lst, filt)


def append_to_keys(adict, preffix):
    """
    @param adict:
    @param preffix:
    @return:
    """
    return {preffix + str(key): (value if isinstance(value, dict) else value)
            for key, value in list(adict.items())}


def append_to_list(lst, preffix):
    """
    @param lst:
    @param preffix:
    @return:
    """
    return [preffix + str(item) for item in lst]


def is_valid_regex(string):
    """
    Checks whether the re module can compile the given regular expression.

    :param string: str

    :return: boolean
    """
    try:
        re.compile(string)
        is_valid = True
    except re.error:
        is_valid = False
    return is_valid


def remove_from_string(string, values):
    """

    :param string:
    :param values:
    :return:
    """
    for v in values:
        string = string.replace(v, '')

    return string


def to_numbers(values, ntype=float, regex=r'\b\d+\b', fill_value=np.NaN,
               remove_symbols=['<', '>', '=']):
    """
    Extracts all numbers in a string, join them and
    transform the result to ntype.

    :param values: list of strings
    List of strings that may contain numbers

    :param ntype: transformation function
    float or int

    :param regex: string
    Regular expression to extract the numbers
    from the strings

    :param fill_value:
    Value to fill the strings that could not
    be converted.

    :param remove_symbols: list of strings
     String values that should be removed of the strings
     before conversion. If None or empty, no filtering
     will be performed.

    :return:
    list of ntypes
    """
    numbers = []
    for v in values:
        try:
            if isinstance(v, str):
                if remove_symbols is not None:
                    v = remove_from_string(v, remove_symbols)

            numbers.append(ntype(v))
        except ValueError:
            if fill_value is not None:
                numbers.append(np.NaN)
            else:
                pass

    return numbers