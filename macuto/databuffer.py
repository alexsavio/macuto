# coding=utf-8
#-------------------------------------------------------------------------------

#Author: Alexandre Manhaes Savio
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU

#License: 3-Clause BSD

#2013, Alexandre Manhaes Savio
#Use this at your own risk!
#-------------------------------------------------------------------------------

import os
import h5py
import tempfile
import numpy as np
import pandas as pd
from pandas import HDFStore
from itertools import product

import logging as log
log.basicConfig(level=log.INFO)


class HdfDataBuffer(object):
    """
    This class is a registry of HDF datasets.
    It maintains one temporary hdf file
    """

    def __init__(self, dir=None, filepath=None, hdf_basepath='/',
                 overwrite_if_exist=False, remove_on_destroy=False):
        """

        :param dir: string

        :param filepath: string

        :param hdf_basepath: string

        :param overwrite_if_exist: bool

        :param remove_on_destroy: bool

        """
        if dir is None:
            dir = tempfile.gettempdir()

        if filepath is None:
            self._fname = self.get_temp_file(dir)
        else:
            self._fname = filepath

        self._fname = self.get_temp_file(dir).name

        self._remove_on_destroy = remove_on_destroy
        self._overwrite = overwrite_if_exist

        self._hdf_basepath = hdf_basepath
        self._hdf_file = None
        self._group = None
        self._datasets = {}

        self.create_hdf_file()

    @staticmethod
    def get_temp_file(dir=None, suffix='.h5'):
        """
        Uses tempfile to create a NamedTemporaryFile using
        the default arguments.

        @param dir: string
        Directory where it must be created.
        If dir is specified, the file will be created
        in that directory, otherwise, a default directory is used.
        The default directory is chosen from a platform-dependent
        list, but the user of the application can control the
        directory location by setting the TMPDIR, TEMP or TMP
        environment variables.

        @param suffix: string
        File name suffix.
        It does not put a dot between the file name and the
        suffix; if you need one, put it at the beginning of suffix.

        @return: file object

        @note:
        Close it once you have used the file.
        """
        return tempfile.NamedTemporaryFile(dir=dir, suffix=suffix)

    def __del__(self):
        """
        Class destructor
        """
        self._hdf_file.close()

        if self._remove_on_destroy:
            os.remove(self._fname)

    def create_hdf_file(self):
        """
        :return: h5py DataSet
        """
        mode = 'w'
        if not self._overwrite and os.path.exists(self._fname):
            mode = 'a'

        self._hdf_file = h5py.File(self._fname, mode)

        if self._hdf_basepath == '/':
            self._group = self._hdf_file['/']
        else:
            self._group = self._hdf_file.create_group(self._hdf_basepath)

    def get_dataset(self, ds_name, mode='r'):
        """
        Returns a h5py dataset given its registered name.

        :param ds_name: string
        Name of the dataset to be returned.

        :return:
        """
        if ds_name in self._datasets:
            return self._datasets[ds_name]
        else:
            return self.create_empty_dataset(ds_name)

    def create_empty_dataset(self, ds_name, dtype=np.float32):
        """
        Creates a Dataset with unknown size.
        Resize it before using.

        :param ds_name: string

        :param dtype: dtype
        Datatype of the dataset

        :return: h5py DataSet
        """
        if ds_name in self._datasets:
            return self._datasets[ds_name]

        try:
            ds = self._group.create_dataset(ds_name, (1, 1), maxshape=None, dtype=dtype)
            self._dataset[ds_name] = ds
            return ds

        except ValueError as ve:
            log.error('Error creating empty dataset ' + ds_name + ' in ' + self._hdf_basepath)
            print(str(ve))

    def create_dataset(self, ds_name, data, attrs={}, dtype=None):
        """
        Saves a Numpy array in a dataset in the HDF file, registers it as
        ds_name and returns the h5py dataset.

        :param ds_name: string
        Registration name of the dataset to be registered.

        :param data: Numpy ndarray

        :param dtype: dtype
        Datatype of the dataset

        :return: h5py dataset
        """
        try:
            if ds_name in self._datasets:
                ds = self._datasets[ds_name]
                if ds.dtype != data.dtype:
                    log.WARN('Dataset and data dtype are different!')
            else:
                if dtype is None:
                    dtype = data.dtype

                ds = self._group.create_dataset(ds_name, data.shape, dtype=dtype)

                if attrs:
                    for key in attrs:
                        setattr(ds.attrs, key, attrs[key])

            ds.read_direct(data)
            self._dataset[ds_name] = ds

        except ValueError as ve:
            log.error('Error creating dataset ' + ds_name + ' in ' + self._hdf_basepath)
            print(str(ve))

        return ds

    def save(self, ds_name, data, dtype=None):
        """
        See create_dataset.
        """
        return self.create_dataset(ds_name, data, dtype)

    @staticmethod
    def force_garbage_collector():
        import gc
        gc.collect()

    @staticmethod
    def get_var_reference_count(p_object):
        import sys
        return sys.getrefcount(p_object)

    @staticmethod
    def get_memory_usage():
        import psutil
        return psutil.virtual_memory()

    @classmethod
    def print_memory_usage(cls):
        print(cls.get_memory_usage())


class NumpyHDFStore(HDFStore):
    """
    dict-like IO interface for storing pandas objects in PyTables
    either Fixed or Table format OR Numpy ndarrays.
    """
    def __init__(self, path, mode=None, complevel=None, complib=None,
                 fletcher32=False, **kwargs):
        super(NumpyHDFStore, self).__init__(path, mode, complevel,
                                            complib, fletcher32, **kwargs)

        self._array_dsname = 'data'

    def _read_array(self, node):
        if hasattr(node, self._array_dsname):
            return getattr(node, self._array_dsname)
        else:
            return node

    @staticmethod
    def _fill_missing_values(df, range_values, fill_value=0, fill_method=None):
        """
        Will get the names of the index colums of df, obtain their ranges from
        range_values dict and return a reindexed version of df with the given
        range values.

        :param df: pandas DataFrame

        :param range_values: dict or array-like
        Must contain for each index column of df an entry with all the values
        within the range of the column.

        :param fill_value: scalar or 'nearest', default 0
        Value to use for missing values. Defaults to 0, but can be any
        "compatible" value, e.g., NaN.
        The 'nearest' mode will fill the missing value with the nearest value in
         the column.

        :param fill_method:  {'backfill', 'bfill', 'pad', 'ffill', None}, default None
        Method to use for filling holes in reindexed DataFrame
        'pad' / 'ffill': propagate last valid observation forward to next valid
        'backfill' / 'bfill': use NEXT valid observation to fill gap

        :return: pandas Dataframe and used column ranges
        reindexed DataFrame and dict with index column ranges
        """
        idx_colnames  = df.index.names

        idx_colranges = [range_values[x] for x in idx_colnames]

        fullindex = pd.Index([p for p in product(*idx_colranges)], name=tuple(idx_colnames))

        fulldf = df.reindex(index=fullindex, fill_value=fill_value, method=fill_method)

        fulldf.index.names = idx_colnames

        return fulldf, idx_colranges

    def get(self, key):
        """
        Retrieve pandas object or group of Numpy ndarrays
        stored in file

        Parameters
        ----------
        key : object

        Returns
        -------
        obj : type of object stored in file
        """
        node = self.get_node(key)
        if node is None:
            raise KeyError('No object named %s in the file' % key)

        if hasattr(node, 'attrs'):
            if 'pandas_type' in node.attrs:
                return self._read_group(node)

        return self._read_array(node)

    def put(self, key, value, attrs={}, format=None, append=False, **kwargs):
        """
        Store object in HDFStore

        Parameters
        ----------
        key : object
        value : {Series, DataFrame, Panel, Numpy ndarray}
        format : 'fixed(f)|table(t)', default is 'fixed'
        fixed(f) : Fixed format
        Fast writing/reading. Not-appendable, nor searchable
        table(t) : Table format
        Write as a PyTables Table structure which may perform
        worse but allow more flexible operations like searching
        / selecting subsets of the data
        append : boolean, default False
        This will force Table format, append the input data to the
        existing.
        encoding : default None, provide an encoding for strings
        """
        if not isinstance(value, np.ndarray):
            super(NumpyHDFStore, self).put(key, value, format, append, **kwargs)
        else:
            group = self.get_node(key)

            # remove the node if we are not appending
            if group is not None and not append:
                self._handle.removeNode(group, recursive=True)
                group = None

            if group is None:
                paths = key.split('/')

                # recursively create the groups
                path = '/'
                for p in paths:
                    if not len(p):
                        continue
                    new_path = path
                    if not path.endswith('/'):
                        new_path += '/'
                    new_path += p
                    group = self.get_node(new_path)
                    if group is None:
                        group = self._handle.createGroup(path, p)
                    path = new_path

            ds_name = kwargs.get('ds_name', self._array_dsname)

            ds = self._handle.createArray(group, ds_name, value)
            if attrs:
                for key in attrs:
                    setattr(ds.attrs, key, attrs[key])

            self._handle.flush()

            return ds

    def _push_dfblock(self, key, df, ds_name, range_values):
        """
        :param key: string
        :param df: pandas Dataframe
        :param ds_name: string
        """
        #create numpy array and put into hdf_file
        vals_colranges = [range_values[x] for x in df.index.names]
        nu_shape = [len(x) for x in vals_colranges]

        return self.put(key, np.reshape(df.values, tuple(nu_shape)),
                        attrs={'axes': df.index.names},
                        ds_name=ds_name, append=True)

    def put_df_as_ndarray(self, key, df, range_values, loop_multiindex=False, unstack=False, fill_value=0, fill_method=None):
        """
        Returns a PyTables HDF Array from df in the shape given by its index columns
        range values.

        :param key: string object

        :param df: pandas DataFrame

        :param range_values: dict or array-like
        Must contain for each index column of df an entry with all the values
        within the range of the column.

        :param loop_multiindex: bool
        Will loop through the first index in a multiindex dataframe, extract a
        dataframe only for one value, complete and fill the missing values and
        store in the HDF.
        If this is True, it will not use unstack.
        This is as fast as unstacking.

        :param unstack: bool
        Unstack means that this will use the first index name to
        unfold the DataFrame, and will create a group with as many datasets
        as valus has this first index.
        Use this if you think the filled dataframe won't fit in your RAM memory.
        If set to False, this will transform the dataframe in memory first
        and only then save it.

        :param fill_value: scalar or 'nearest', default 0
        Value to use for missing values. Defaults to 0, but can be any
        "compatible" value, e.g., NaN.
        The 'nearest' mode will fill the missing value with the nearest value in
         the column.

        :param fill_method:  {'backfill', 'bfill', 'pad', 'ffill', None}, default None
        Method to use for filling holes in reindexed DataFrame
        'pad' / 'ffill': propagate last valid observation forward to next valid
        'backfill' / 'bfill': use NEXT valid observation to fill gap

        :return: PyTables data node
        """
        idx_colnames = df.index.names
        #idx_colranges = [range_values[x] for x in idx_colnames]

        #dataset group name if not given
        if key is None:
            key = idx_colnames[0]

        if loop_multiindex:
            idx_values = df.index.get_level_values(0).unique()

            for idx in idx_values:
                vals, _ = self._fill_missing_values(df.xs((idx,), level=idx_colnames[0]),
                                                    range_values,
                                                    fill_value=fill_value,
                                                    fill_method=fill_method)

                ds_name = str(idx) + '_' + '_'.join(vals.columns)

                self._push_dfblock(key, vals, ds_name, range_values)

            return self._handle.get_node('/' + str(key))

        #separate the dataframe into blocks, only with the first index
        else:
            if unstack:
                df = df.unstack(idx_colnames[0])
                for idx in df:
                    vals, _ = self._fill_missing_values(df[idx], range_values,
                                                        fill_value=fill_value,
                                                        fill_method=fill_method)
                    vals = np.nan_to_num(vals)

                    ds_name = '_'.join([str(x) for x in vals.name])

                    self._push_dfblock(key, vals, ds_name, range_values)

                return self._handle.get_node('/' + str(key))

        #not separate the data
        vals, _ = self._fill_missing_values(df, range_values,
                                            fill_value=fill_value,
                                            fill_method=fill_method)

        ds_name = self._array_dsname

        return self._push_dfblock(key, vals, ds_name, range_values)
