
from itertools import combinations


def treefall(iterable):
    """
    Generate all combinations of the elements of iterable and its subsets.

    Parameters
    ----------
    iterable: list, set or dict or any iterable object

    Returns
    -------
    A generator of all possible combinations of the iterable.

    Example:
    -------
    >>> for i in treefall([1, 2, 3, 4, 5]): print(i)
    >>> (1, 2, 3)
    >>> (1, 2)
    >>> (1, 3)
    >>> (2, 3)
    >>> (1,)
    >>> (2,)
    >>> (3,)
    >>> ()
    """
    num_elems = len(iterable)
    for i in range(num_elems, -1, -1):
        for c in combinations(iterable, i):
            yield c
