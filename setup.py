#!/usr/bin/python

import os
import ast
import configparser
from distutils.core import setup

def ast_dict_items(adict):
    """
    @param adict:
    @return:
    """
    for i in adict:
        try:
            adict[i] = ast.literal_eval(adict[i])
        except ValueError as ve:
            print('Error trying to evaluate expression: ' + adict[i])
            print(str(ve))

    return adict


def get_config_dictionary(config_file='setup.cfg', sections=['metadata',
                                                             'global',
                                                             'files']):
    """
    @param config_file: string
     Path to config file.
     Content style should be:
     http://docs.python.org/3.3/library/configparser.html

    @param sections:
     Name of the config sections to be parsed into
     the returned dictionary

    @return: dict

    """
    config = configparser.ConfigParser()
    with open('setup.cfg') as f:
        config.read_file(f)

    cfgdict = {}
    for s in sections:
        sdict = {}
        try:
            sdict = dict(config[s])

        except KeyError as ke:
            print('Could not find section: ' + s +
                  ' in file: ' + os.abspath(config_file))
            print(str(ke))

        cfgdict.update(ast_dict_items(sdict))

    return cfgdict


if __name__ == '__main__':
    setup(**get_config_dictionary())