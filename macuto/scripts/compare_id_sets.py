#!/usr/bin/env python

#TODO
#Transform this into a CLI

import re
import os
import logging
import collections
import numpy as np

from tabulate import tabulate
from dcm_anonymize import get_all_patient_mri_ids

#logging config
logging.basicConfig(level=logging.DEBUG, filename='idset_comparator.log',
                    format="%(asctime)-15s %(message)s")
log = logging.getLogger('idset_comparing')

class idset(np.ndarray):
    """
    Array of identifiers.

    ...

    Attributes
    ----------
    name: string
    Name of the set

    Methods
    -------
    get_repetitions()

    print_counts()

    print_unique_nums()
        Prints the total number of IDs and the number of unique IDs

    """

    #@classmethod
    def __new__(cls, input_array, name=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.name = name

        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see idset.__array_finalize__ for comments
        if obj is None: return
        self.name = getattr(obj, 'name', None)

    def get_repetitions(self):
        return [i for i in np.unique(self) if i is not None and np.sum(self == i) > 1]

    def print_unique_nums(self):
        table = []
        table.append(['Number of {0}: '.format(self.name), len(self)])
        table.append(['Unique: ', len(np.unique(self))])
        print(tabulate(table))

    def print_counts(self):
        reps = self.get_repetitions()
        table = [[self.name, 'Repetitions']]
        for rep in reps:
            table.append([rep, np.sum(self == rep)])
            #print('{0} {1} appears {2} times'.format(self.name, rep,
            #                                         np.sum(self == rep)))

        print(tabulate(table, headers='firstrow'))

    def self_test(self):
        print('======================================================')
        print('Checking {0} values:'.format(self.name))
        self.print_unique_nums()
        print('\n')
        self.print_counts()


class idset_with_reference(idset):
    """
    Array of identifiers with a list of references for each ID.
    The list of references and IDs must have the same number,
    we assume they are in the same order.
    ...

    Attributes
    ----------
    name: string
    Name of the set

    reflst: list or ndarray
    List of references for each ID

    refname: string
    Name of the reference

    Methods
    -------
    print_repeated_references()

    """
    #@classmethod
    def __new__(cls, input_array, name=None, reflst=None, refname=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = idset.__new__(cls, input_array, name)
        obj.reflst = reflst
        obj.refname = refname

        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see idset.__array_finalize__ for comments
        if obj is None: return
        super(idset_with_reference, self).__array_finalize__(obj)

        self.reflst = getattr(obj, 'reflst', None)
        self.refname = getattr(obj, 'refname', None)

    def print_repeatedid_references(self):
        reps = self.get_repetitions()
        refs = np.array(self.reflst)

        for rep in reps:
            if rep is None:
                continue

            try:
                if rep == np.array(None).astype(self.dtype):
                    continue
            except:
                pass

            table = []
            for name in refs[np.where(self == rep)]:
#   print('{0} {1} corresponds to {2} {3}'.format(self.name.capitalize(), rep,
#                                                 self.refname.lower(), name))
                table11 = ''
                table12 = name
                if not table:
                    table11 = '{0} {1} corresponds to {2}'.format(self.name,
                                                                  rep,
                                                                  self.refname)
                table.append([table11, table12])

            print(tabulate(table))


    def get_noneid_references(self):
        """
        Returns
        -------
        ndarray
        Array of references in self.reflst whose self id is None.
        """
        #return [self.reflst[idx] for idx, idval in enumerate(self) if idval is None]
        try:
            nun = np.array(None).astype(self.dtype)
            return np.array(self.reflst)[self == nun]
        except:
            nun = None
            return np.array(self.reflst)[self is None]

    def print_noneid_references(self):
        nune_refs = self.get_noneid_references()
        table = [['Has {0} as None'.format(self.name), '']]
        for ref in nune_refs:
            table.append(['', ref])
            #print('{0} {1} has {2} as None.'.format(self.refname, ref,
            #                                        self.name))
        if len(table) > 1:
            print(tabulate(table))

    def self_test(self):
        super(idset_with_reference, self).self_test()
        print('\n')
        self.print_repeatedid_references()
        print('\n')
        self.print_noneid_references()


class idset_comparator(collections.OrderedDict):
    """

    """
    def __init__(self):
        super(idset_comparator, self).__init__()

    @staticmethod
    def _get_elem(list, idx, default=None):
        elem = default
        try:
            elem = list[idx]
        except:
            pass
        return elem

    @staticmethod
    def _tabulate_2_lists(list1, list2):
        """
        """
        tablst = []
        for idx in list(range(max(len(list1), len(list2)))):
            elem1 = idset_comparator._get_elem(list1, idx, '')
            elem2 = idset_comparator._get_elem(list2, idx, '')

            tablst.append([elem1, elem2])
        return tablst

    @staticmethod
    def _tabulate_4_lists(list1, list2, list3, list4):
        """
        """
        tablst = []
        for idx in list(range(max(len(list1), len(list2),
                                  len(list3), len(list4)))):
            elem1 = idset_comparator._get_elem(list1, idx, '')
            elem2 = idset_comparator._get_elem(list2, idx, '')
            elem3 = idset_comparator._get_elem(list3, idx, '')
            elem4 = idset_comparator._get_elem(list4, idx, '')

            tablst.append([elem1, elem2, elem3, elem4])
        return tablst

    def _print_general_vs_table(self, idset1, idset2):
        """
        :param idset1:
        :param idset2:
        """
        ref1name = ''
        set1_hasref = isinstance(idset1, idset_with_reference)
        if set1_hasref:
            ref1arr = np.array(idset1.reflst)
            ref1name = idset1.refname

        ref2name = ref1name
        set2_hasref = isinstance(idset2, idset_with_reference)
        if set2_hasref:
            ref2arr = np.array(idset2.reflst)
            ref2name = idset2.refname
        else:
            ref2name = ref1name

        #First show a general table
        hdr11 = '{0} > {1}'.format(idset1.name, idset2.name)
        hdr12 = '{0} > {1} {2}'.format(idset1.name, idset2.name, ref2name)
        hdr13 = '{0} < {1}'.format(idset1.name, idset2.name)
        hdr14 = '{0} < {1} {2}'.format(idset1.name, idset2.name, ref1name)
        table = [[hdr11, hdr12, hdr13, hdr14]]

        set1 = set(idset1)
        set2 = set(idset2)
        row11 = list(set1 - set2)
        if set1_hasref:
            row12 = [ref1arr[np.where(idset1 == nom)][0] for nom in row11]
        else:
            row12 = ['Not found' for _ in row11]

        row13 = list(set2 - set1)
        if set2_hasref:
            row14 = [ref2arr[np.where(idset2 == nom)][0] for nom in row13]
        else:
            row14 = ['Not found' for _ in row13]

        tablst = self._tabulate_4_lists(row11, row12, row13, row14)
        table.extend(tablst)

        if len(table) > 1:
            print(tabulate(table, headers='firstrow'))
            print('\n')

    def _print_foreign_repetition_table(self, idset1, idset2):
        """
        :param idset1:
        :param idset2:
        """

        assert(isinstance(idset1, idset_with_reference))
        assert(isinstance(idset2, idset))

        reps = idset2.get_repetitions()
        if len(reps) < 1:
            return

        refs = np.array(idset1.reflst)
        table = [['{0} {1} values of repetitions in {2}'.format(idset1.name,
                                                                idset1.refname,
                                                                idset2.name),
                  '']]

        for rep in reps:
            if np.any(idset1 == rep):
                matches = refs[np.where(idset1 == rep)]
                myrep = rep
                for m in matches:
                    table.append([myrep, m])
                    myrep = ''

        print(tabulate(table, headers='firstrow'))
        print('\n')

    def print_compare_idsets(self, idset1_name, idset2_name):
        """
        """
        try:
            idset1 = self[idset1_name]
            idset2 = self[idset2_name]
        except KeyError as ke:
            log.error('Error compare_idsets: getting keys {0} and {1}'.format(idset1_name,
                                                                              idset2_name))
            import sys, pdb
            pdb.post_mortem(sys.exc_info()[2])
            raise

        assert(isinstance(idset1, idset))
        assert(isinstance(idset2, idset))

        hdr11 = '{0} > {1}'.format(idset1_name, idset)
        hdr12 = '{0} < {1}'.format(idset1_name, idset2_name)
        table = [[hdr11, hdr12]]

        set1 = set(idset1)
        set2 = set(idset2)
        row11 = list(set1 - set2)
        row12 = list(set2 - set1)

        tablst = self._tabulate_2_lists(row11, row12)
        table.extend(tablst)

        if len(table) > 1:
            print(tabulate(table, headers='firstrow'))
            print('\n')

    def print_compare_idsets_one_ref(self, idset1_name, idset2_name):
        """
        idset1_name: string
        key of an idset_with_reference

        idset2_name: string
        key of an idset
        """
        try:
            idset1 = self[idset1_name]
            idset2 = self[idset2_name]
        except KeyError as ke:
            log.error('Error compare_idsets: getting keys {0} and {1}'.format(idset1_name,
                                                                              idset2_name))
            import sys, pdb
            pdb.post_mortem(sys.exc_info()[2])
            raise

        assert(isinstance(idset1, idset_with_reference))
        assert(isinstance(idset2, idset))

        self._print_general_vs_table(idset1, idset2)
        self._print_foreign_repetition_table(idset1, idset2)

    def print_compare_idsets_two_refs(self, idset1_name, idset2_name):
        """
        idset1_name: string
        key of an idset_with_reference

        idset2_name: string
        key of an idset
        """
        try:
            idset1 = self[idset1_name]
            idset2 = self[idset2_name]
        except KeyError as ke:
            log.error('Error compare_idsets: getting keys {0} and {1}'.format(idset1_name,
                                                                              idset2_name))
            import sys, pdb
            pdb.post_mortem(sys.exc_info()[2])
            raise

        assert(isinstance(idset1, idset_with_reference))
        assert(isinstance(idset2, idset_with_reference))

        self._print_general_vs_table(idset1, idset2)
        self._print_foreign_repetition_table(idset1, idset2)
        self._print_foreign_repetition_table(idset2, idset1)

    def print_all_comparisons(self):
        """
        """
        keys = self.keys()
        for idx, set1 in enumerate(keys):
            for set2 in keys[idx+1:]:
                if set1 == set2:
                    continue

                is_idset1 = isinstance(self[set1], idset)
                is_idset2 = isinstance(self[set2], idset)
                if not is_idset1 or not is_idset2:
                    continue

                print('======================================================')
                print('{0} VS. {1}'.format(set1, set2))
                print('======================================================')

                is_idset1 = isinstance(self[set1], idset) and not isinstance(self[set1], idset_with_reference)
                is_idset2 = isinstance(self[set2], idset) and not isinstance(self[set2], idset_with_reference)
                if is_idset1 and is_idset2:
                    self.print_compare_idsets(set1, set2)

                is_refidset1 = isinstance(self[set1], idset_with_reference)
                is_refidset2 = isinstance(self[set2], idset_with_reference)
                if is_refidset1 and not is_refidset2:
                    self.print_compare_idsets_one_ref(set1, set2)

                elif is_refidset1 and is_refidset2:
                    self.print_compare_idsets_two_refs(set1, set2)

                print('======================================================')
                print('\n')


if __name__ == '__main__':

    curdir = '/data/santiago/data'

    curdir = os.path.abspath(curdir)
    subjlst = os.listdir(curdir)
    subjlst.sort()

    #get IDs from folder names
    idregex = r'[N|P]?\d\d*-?\d?$'
    dirids = [re.search(idregex, i).group(0) for i in subjlst]

    #print DICOM ids found for each subject
    enhe = '\xd1'
    all_dicids = collections.OrderedDict((i, get_all_patient_mri_ids(os.path.join(curdir, i))) for i in subjlst)
    idtab = [[k.replace(enhe, 'N'), str(list(all_dicids[k]))] for k in all_dicids.keys()]
    print(tabulate(idtab, headers=['Folder Name', 'DICOM IDs']))

    #get IDs from DICOM files, if there are more than one,
    #looks for the one that matches dirids[i]
    dicids = []
    for idx, i in enumerate(subjlst):
        sids = all_dicids[i]
        if len(sids) == 1:
            sid = sids.pop()
        elif len(sids) > 1:
            for sid in sids:
                if sid == dirids[idx]:
                    break
        else:
            sid = 'None'
        dicids.append(sid)

    csvids = ['N122054', '99100167', '99106030', '886783', '99109169',
              'P99106025', '99106027', 'N100712', 'P99117222', 'N640001',
              '1560237', '697987', '99135286', 'N859000', 'N640000', 'N968000',
              'N890000', '55826', 'P99143948', '1542604', '1325097', 'N110626',
              '472315', 'N092449', 'N328000', 'N115611', '1113611', '49612',
              '261676', '26513', '734615', '890405', '687685', '1328215',
              '15817', '472717', '1365535', '921735', '470663', '1084809',
              '470848', '1578187', 'N104950', '381249', '320712', '99130021',
              '99135285', '99130019', '1124286', 'N265000', '1740968',
              '99141533', '886153', '238394', 'N102326', 'N140000', 'N031000',
              '375000', 'N781000', 'N111637', 'N8590001', 'N750000', 'N406000',
              'N875000', 'N718000', '1954086', 'N281000', '1942888', '1328924',
              '1756647', '241599', '1313224', 'N103159', '2136656', '1969022',
              '1549266', '99072462', '1983029', 'N437000', '661449', '24525',
              '683479', '248966', '1939978', '1547734', '35207', '1738856',
              '1164737', '671898', '1314109', '1770861', '99135289', 'N718000',
              'N312000', '45706', '99105995', '99109170', '46689', '400742',
              'N093000', '99117888', '99117812', '925987', '99119960',
              '482470', 'N156000',  '707383', '1952854', '685906', '66334',
              'N173253', 'N133209', '1554723', '2233682', '1956804', '19797',
              '936976', '1539140', '96049', '458039', '1364867', '1335553']


    #remove a nasty spanish symbol
    enhe = '\xd1'
    subjlst = [s.replace(enhe, 'N') for s in subjlst]

    #Set of IDs from folder names
    diridset = idset_with_reference(dirids, name='Folder ID',
                                    reflst=subjlst, refname='Folder')

    #Set of IDs from DICOM data
    dicidset = idset_with_reference(dicids, name='DICOM ID',
                                    reflst=subjlst, refname='Folder')

    csvidset = idset(csvids, name='CSV ID')

    diridset.self_test()
    dicidset.self_test()
    csvidset.self_test()

    idcomp = idset_comparator()
    idcomp[diridset.name] = diridset
    idcomp[dicidset.name] = dicidset
    idcomp[csvidset.name] = csvidset

    idcomp.print_all_comparisons()

