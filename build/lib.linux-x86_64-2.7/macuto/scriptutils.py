import inspect
# functions


def whoami():
    """Get the name of the current function"""
    return inspect.stack()[1][3]


def whosdaddy():
    """Get the name of the current function"""
    return inspect.stack()[2][3]
