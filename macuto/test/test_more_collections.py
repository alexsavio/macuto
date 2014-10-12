# -*- coding: utf-8 -*-

from collections import OrderedDict
from macuto.more_collections import merge_dict_of_lists


def test_merge_dict_of_lists():

    mamas = OrderedDict([('a', [1, 2, 3, 4, 5]),
                         ('b', [6, 7, 8, 9, 10]),
                         ('c', [11, 12, 13, 14, 15]),
                         ('d', [16, 17, 18, 19, 20])])

    indices = ([0], [3])

    result = OrderedDict([('a', [1, 2, 3, 4, 5, 16, 17, 18, 19, 20]),
                          ('b', [6, 7, 8, 9, 10]),
                          ('c', [11, 12, 13, 14, 15])])

    assert(merge_dict_of_lists(mamas, indices) == result)
