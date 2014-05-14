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


'''
Folder Name                                        DICOM IDs
-------------------------------------------------  --------------------------
ABERASTURI_55826                                   ['455826']
AJURIA_PUJANA_EMILIA1__46689                       ['46689']
ALVAREZ_GONZALEZ_FLORENCIO__1325097                ['1325097']
ALVARO_ALVARO_LUIS__96049                          ['96049']
AMARO_MATIAS_SEBASTIAN__P99145133                  ['P99145133']
ANCIN_JUARROS_RAFAEL__1364867                      ['1364867']
ANDRES MARTINEZ JUAN JACINTO N133209               ['935725']
ANDRES_CASTA_EDA_INMACULADA__734615                ['734615']
ANGULO_REVILLAS_FELISA1__2233682                   ['2233682']
APILANEZ_LOPEZ_JULIA__1954086                      ['1954086']
ARANZABAL_CERECERA_AURORA__886153                  ['886153']
ARJONA_SANCHEZ_AGUSTIN__1539140                    ['1539140']
ARNAIZ_ETXEBARRIETA_BEGO_A__1542604                ['1542604']
ARRABAL_SOTA_MARIA_MERCEDES__1560237               ['1560237']
ARRASTIO_URIARTE_RAQUEL__45706                     ['45706']
ARRAUSI_HERNANDO_JOSEFA_VITORIANA__238394          ['238394']
ASPE_RODRIGO_ESTHER__99109167                      ['99109167']
AZCARATE_SAN_JOSE_MARIA_CARMEN__1313224            ['1313224']
BARRIO CUENDE N437000                              ['3201903']
BARRIOS_FIDALGO_JULIANA__921735                    ['921735']
BARRON_MARTINEZ_DE_OSABA_ANTONINO__P99117222       ['P99117222', '99117222']
BA_ALES_BASARRATE_RAMIRO__1740968                  ['1740968']
BELTRAN_DE_HEREDIA_ELIZONDO_VICTORIA__261676       ['261676']
BELTRAN_UZABAL_BLANCA__1328215                     ['1328215']
BLANCO_GARCIA_NIEVES__P99143948                    ['P99143948']
BORINAGA_PEREZ_DE_NANCLARES_LEONOR__1770861        ['1770861']
CALVO_ESTEBAN_MARIA_CARMEN__482470                 ['482470', '1995804']
CASADO_CUBEROS_DOLORES__1942888                    ['1942888']
CENTENO_GONZALEZ_JOSE_LUIS__1549266                []
CORCUERA_RUIZ_DE_AUSTRI_ROSARIO__661449            ['661449']
CRUZ_CUESTA_JOSE_MANUEL__1365535                   ['1365535']
CRUZ_MOLINA_FRANCISCA__400742                      ['400742']
CUETO_RODRIGUEZ_FRANCISCO__35207                   ['35207']
DELGADO_NAHARRO_ISIDRO__99109170                   ['99109170', '1278474']
DE_BLAS_ABAD_MARIA__99135285                       ['99135285', '1092939']
DE_LA_ROSA_SAN_JOSE_DIONISIO__381249               ['381249']
DIAZ SAENZ 683479                                  ['683479']
DIEZ_GONZALEZ_MARIA_LUISA__19797                   ['19797']
DURANA_PEREZ_DE_HEREDIA_ROSARIO__99109169          ['99109169']
DURAN_RODRIGUEZ_TEODORO__99105995                  ['P99105995', '99105995']
DURAN_VINAGRE_MARIA_ANTONIA__99117812              ['99117812']
FERNANDEZ BETONO N843000                           ['1341577']
FERNANDEZ IBANEZ N084248                           ['P99148727']
FERNANDEZ_DE_ARROYABE_PAGOAGA_JESUS__99130019      ['99130019']
FERNANDEZ_DE_MATAUCO_MARTINEZ_VALENTIN__1328924    ['1328924']
FERNANDEZ_DE_ZA_ARTU_RUIZ_DE_E_LUIS__890405        ['890405']
GALLEGO MATIN 24525                                ['24525']
GAMARRA MAYOR N110626                              ['39712']
GANZABAL DIONISIO N140000                          ['1312383']
GARCIA GUTIERREZ ILDEFONSO N093000                 ['99074553', '1561125']
GARCIA PARRO N173253                               ['280364']
GARCIA PORTAL 1983029                              ['1983029']
GARCIA_DE_CORTAZAR_OCHOA_DE_EC_ELISA__1314109      ['1314109']
GARCIA_DE_SALAZAR_GOMEZ_DE_BAL_HORTENSIA__1939978  ['1939978']
GAVI_A_GONZALEZ_ROSARIO__1756647                   ['1756647']
GIL ROSCO FERNANDA N406000                         ['238542']
GONZALEZ_ALVAREZ_JOSE_ANTONIO__99135286            ['99135286']
GONZALEZ_PLAZA_CONCHA__470663                      ['470663']
GUERRERO RAFAEL N156000                            ['53144']
GUTIERREZ_RODRIGUEZ_JULIA__99106027                ['99106027']
HERNANGOMEZ_GORGOJO_ISABEL__1547734                ['1547734']
HERRERA_CALDERON_LUCAS__671898                     ['671898']
HERREROS_MARA_ON_ROBERTO__2136656                  ['2136656']
INSAGURBE CATALINA N031000                         ['22403']
IRASTORZA CARMELO N265000                          ['P99145371']
I_IGO_LOPEZ_DE_GUERE_U_AMELIA__99130021            ['99130021']
JIMENEZ_LOPEZ_DE_LACALLE_JESUS__26513              ['26513']
LERCHUNDI_DE_ECIOLAZA_MARIA_LUISA__1084809         ['1084809']
LINARES FREIRE N859000                             ['1543368']
LOPEZ DI CASTILLO ALFREDO N640000                  ['1327760']
LOPEZ_DE_ARROYABE_GUEVARA_MARIA_PILAR__470848      ['470848']
LOPEZ_DE_MUNAIN_RUIZ_ARECHAVAL_NICOLAS__241599     ['241599']
LOPO RUANO N640000                                 ['99012114']
LUQUE OSUNA N328000                                ['472388']
MARTINEZ LAHOZ LUCIA 687685                        ['687685']
MARTIN_RAMIRO_JUAN__99117888                       ['99117888']
MENDO BAZAN CANDELA 375000                         ['1522450']
MERIDA_TOVAR_FIDELA__49612                         ['49612']
MOLINERO_DUE_AS_JOSE_LUIS__925987                  ['925987']
MONTOYA ALFONSO N781000                            ['1519562']
MUGICA_OSTIZ_JOSE_RAMON__697987                    []
MU_OZ_ARGOMANIZ_ANA_MARIA__1578187                 ['1578187']
MU_OZ_RAEZ_MARIA_DEL_CARMEN__1164737               ['1164737']
NOVAS_GONZALEZ_SERVANDO__685906                    ['685906']
OCIO_SAMANIEGO_MARIA_JESUS__99106030               []
OLLOQUI_RUEDA_ANTONIO__15817                       ['15817']
ONAINDIA N750000                                   ['444802']
ORTIZ_DE_GUINEA_ALDAY_FRANCISCO_JAVIER__248966     ['248966']
OTI  AJA ANICETO N859000-1                         ['1517716']
ONEDERRA N111637                                   ['1547802']
PARAMO_BARRIO_MANUEL__1113611                      ['1113611']
PEREZ FERNANDEZ DE LANDA N122054                   ['99109164']
PORRAS_DOMINGO_ATILANO__P99106025                  ['P99117223', 'P99106025']
RODRIGUEZ REVUELTA N092449                         ['P292762']
RODRIGUEZ_POMBAR_JOAQUIN__1969022                  ['1969022']
ROMERO_CHAMORRO_ANTONIO__1738856                   ['1738856']
ROMERO_GIL_DE_ZU_IGA_FRANCISCO__66334              ['66334']
ROPERO SANTIAGO N718000                            ['99143081']
RUIZ DE ARBULO JUANA N100712                       ['1965124']
RUIZ DE EGUINO N102326                             ['1738381']
RUIZ LARRINAGA N875000                             ['896178']
RUIZ RODRIGUEZ EXPECTACION N718000                 ['P99148726']
SAENZ CAMARA LA  N115611                           ['1548357']
SANCHA CALABOZO 936976                             ['936976']
SANTA_CRUZ_HERNANDO_CELESTINO__1952854             ['1952854']
SANTIAGO_AGUAYO_ALBERTO__320712                    ['1511222', '320712']
SAN_PEDRO_RUIZ_DE_VI_ASPRE_ISABEL__1554723         ['1554723']
SEDANO_MARTINEZ_JESUS__1956804                     ['1956804']
SORIA_MAQUEDA_AMANCIO__458039                      ['458039']
TORNOS_RUIZ_ENCARNACION__1124286                   []
TORRALBA_FERNANDEZ_GREGORIO__1335553               ['1335553']
TRINIDAD_GONZALEZ_FELIX__707383                    ['707383']
UGARTE ENRIQUE N312000                             ['P99090486']
UGARTONDO JM N104950                               ['1529071']
VALDECANTOS M TERESA N968000                       ['P312430']
VALDECANTOS MARINA N890000                         ['229418']
VAZQUEZ_BELATEGUI_MARIA_CONCEPCION__886783         ['886783']
VELAZQUEZ_MORENO_LUIS__99135289                    ['99135289', 'P99135289']
VICIOSO_BRETON_ROSARIO__99119960                   ['1574185', '99119960']
VIEDMA_MEDINA_RAMONA__472717                       ['472717']
VILLANUEVA M JESUS N281000                         ['10102873']
VINASPRE GARCIA l N103159                          ['1111649']
ZUBILLAGA_ALTUBE_FERNANDO__99072462                ['99072462']
ZURBITU_ANGULO_MARIA_BEGO_A__472315                ['472315']
======================================================
Checking Folder ID values:
--------------------  ---
Number of Folder ID:  124
Unique:               122
--------------------  ---


Folder ID      Repetitions
-----------  -------------
N640000                  2
N718000                  2


---------------------------------------  ---------------------------------
Folder ID N640000 corresponds to Folder  LOPEZ DI CASTILLO ALFREDO N640000
                                         LOPO RUANO N640000
---------------------------------------  ---------------------------------
---------------------------------------  ----------------------------------
Folder ID N718000 corresponds to Folder  ROPERO SANTIAGO N718000
                                         RUIZ RODRIGUEZ EXPECTACION N718000
---------------------------------------  ----------------------------------


======================================================
Checking DICOM ID values:
-------------------  ---
Number of DICOM ID:  124
Unique:              121
-------------------  ---


DICOM ID      Repetitions
----------  -------------
None                    4




--------------------  ------------------------------------
Has DICOM ID as None
                      CENTENO_GONZALEZ_JOSE_LUIS__1549266
                      MUGICA_OSTIZ_JOSE_RAMON__697987
                      OCIO_SAMANIEGO_MARIA_JESUS__99106030
                      TORNOS_RUIZ_ENCARNACION__1124286
--------------------  ------------------------------------
======================================================
Checking CSV ID values:
-----------------  ---
Number of CSV ID:  122
Unique:            121
-----------------  ---


CSV ID      Repetitions
--------  -------------
N718000               2
======================================================
Folder ID VS. DICOM ID
======================================================
Folder ID > DICOM ID    Folder ID > DICOM ID Folder           Folder ID < DICOM ID    Folder ID < DICOM ID Folder
----------------------  ------------------------------------  ----------------------  ------------------------------------
N843000                 FERNANDEZ BETONO N843000              1517716                 OTI  AJA ANICETO N859000-1
N781000                 MONTOYA ALFONSO N781000               10102873                VILLANUEVA M JESUS N281000
N111637                 ONEDERRA N111637                      1548357                 SAENZ CAMARA LA  N115611
N750000                 ONAINDIA N750000                      1327760                 LOPEZ DI CASTILLO ALFREDO N640000
N092449                 RODRIGUEZ REVUELTA N092449            1522450                 MENDO BAZAN CANDELA 375000
N437000                 BARRIO CUENDE N437000                 99143081                ROPERO SANTIAGO N718000
N718000                 ROPERO SANTIAGO N718000               1561125                 GARCIA GUTIERREZ ILDEFONSO N093000
N103159                 VINASPRE GARCIA l N103159             1529071                 UGARTONDO JM N104950
N312000                 UGARTE ENRIQUE N312000                P292762                 RODRIGUEZ REVUELTA N092449
N100712                 RUIZ DE ARBULO JUANA N100712          22403                   INSAGURBE CATALINA N031000
N875000                 RUIZ LARRINAGA N875000                P99145371               IRASTORZA CARMELO N265000
N102326                 RUIZ DE EGUINO N102326                39712                   GAMARRA MAYOR N110626
N281000                 VILLANUEVA M JESUS N281000            1543368                 LINARES FREIRE N859000
N406000                 GIL ROSCO FERNANDA N406000            238542                  GIL ROSCO FERNANDA N406000
N122054                 PEREZ FERNANDEZ DE LANDA N122054      1341577                 FERNANDEZ BETONO N843000
N104950                 UGARTONDO JM N104950                  455826                  ABERASTURI_55826
697987                  MUGICA_OSTIZ_JOSE_RAMON__697987       P312430                 VALDECANTOS M TERESA N968000
N133209                 ANDRES MARTINEZ JUAN JACINTO N133209  None                    CENTENO_GONZALEZ_JOSE_LUIS__1549266
N859000-1               OTI  AJA ANICETO N859000-1            1965124                 RUIZ DE ARBULO JUANA N100712
N265000                 IRASTORZA CARMELO N265000             1312383                 GANZABAL DIONISIO N140000
N093000                 GARCIA GUTIERREZ ILDEFONSO N093000    P99148727               FERNANDEZ IBANEZ N084248
N173253                 GARCIA PARRO N173253                  P99148726               RUIZ RODRIGUEZ EXPECTACION N718000
N859000                 LINARES FREIRE N859000                229418                  VALDECANTOS MARINA N890000
N115611                 SAENZ CAMARA LA  N115611              444802                  ONAINDIA N750000
N110626                 GAMARRA MAYOR N110626                 896178                  RUIZ LARRINAGA N875000
N140000                 GANZABAL DIONISIO N140000             99012114                LOPO RUANO N640000
55826                   ABERASTURI_55826                      1111649                 VINASPRE GARCIA l N103159
N968000                 VALDECANTOS M TERESA N968000          935725                  ANDRES MARTINEZ JUAN JACINTO N133209
1124286                 TORNOS_RUIZ_ENCARNACION__1124286      280364                  GARCIA PARRO N173253
N640000                 LOPEZ DI CASTILLO ALFREDO N640000     P99090486               UGARTE ENRIQUE N312000
N084248                 FERNANDEZ IBANEZ N084248              1547802                 ONEDERRA N111637
N156000                 GUERRERO RAFAEL N156000               472388                  LUQUE OSUNA N328000
375000                  MENDO BAZAN CANDELA 375000            53144                   GUERRERO RAFAEL N156000
N031000                 INSAGURBE CATALINA N031000            99109164                PEREZ FERNANDEZ DE LANDA N122054
N328000                 LUQUE OSUNA N328000                   1738381                 RUIZ DE EGUINO N102326
N890000                 VALDECANTOS MARINA N890000            3201903                 BARRIO CUENDE N437000
1549266                 CENTENO_GONZALEZ_JOSE_LUIS__1549266   1519562                 MONTOYA ALFONSO N781000
99106030                OCIO_SAMANIEGO_MARIA_JESUS__99106030










======================================================


======================================================
Folder ID VS. CSV ID
======================================================
Folder ID > CSV ID    Folder ID > CSV ID Folder          Folder ID < CSV ID    Folder ID < CSV ID Folder
--------------------  ---------------------------------  --------------------  ---------------------------
N843000               FERNANDEZ BETONO N843000           99100167              Not found
99109167              ASPE_RODRIGO_ESTHER__99109167      99141533              Not found
N084248               FERNANDEZ IBANEZ N084248           N640001               Not found
P99145133             AMARO_MATIAS_SEBASTIAN__P99145133  N8590001              Not found
N859000-1             OTI  AJA ANICETO N859000-1


Folder ID Folder values of repetitions in CSV ID
--------------------------------------------------  ----------------------------------
N718000                                             ROPERO SANTIAGO N718000
                                                    RUIZ RODRIGUEZ EXPECTACION N718000


======================================================


======================================================
DICOM ID VS. CSV ID
======================================================
DICOM ID > CSV ID    DICOM ID > CSV ID Folder              DICOM ID < CSV ID    DICOM ID < CSV ID Folder
-------------------  ------------------------------------  -------------------  --------------------------
1517716              OTI  AJA ANICETO N859000-1            N781000              Not found
10102873             VILLANUEVA M JESUS N281000            N111637              Not found
1738381              RUIZ DE EGUINO N102326                N750000              Not found
1548357              SAENZ CAMARA LA  N115611              N092449              Not found
1327760              LOPEZ DI CASTILLO ALFREDO N640000     N437000              Not found
1522450              MENDO BAZAN CANDELA 375000            N718000              Not found
99143081             ROPERO SANTIAGO N718000               N103159              Not found
1561125              GARCIA GUTIERREZ ILDEFONSO N093000    N312000              Not found
P99145133            AMARO_MATIAS_SEBASTIAN__P99145133     N100712              Not found
1529071              UGARTONDO JM N104950                  N875000              Not found
P292762              RODRIGUEZ REVUELTA N092449            N102326              Not found
22403                INSAGURBE CATALINA N031000            99100167             Not found
P99145371            IRASTORZA CARMELO N265000             N281000              Not found
39712                GAMARRA MAYOR N110626                 N406000              Not found
1543368              LINARES FREIRE N859000                N122054              Not found
238542               GIL ROSCO FERNANDA N406000            N104950              Not found
1341577              FERNANDEZ BETONO N843000              697987               Not found
455826               ABERASTURI_55826                      N133209              Not found
P312430              VALDECANTOS M TERESA N968000          N265000              Not found
None                 CENTENO_GONZALEZ_JOSE_LUIS__1549266   N093000              Not found
1965124              RUIZ DE ARBULO JUANA N100712          N173253              Not found
1312383              GANZABAL DIONISIO N140000             N859000              Not found
P99148727            FERNANDEZ IBANEZ N084248              N115611              Not found
P99148726            RUIZ RODRIGUEZ EXPECTACION N718000    N110626              Not found
229418               VALDECANTOS MARINA N890000            N140000              Not found
444802               ONAINDIA N750000                      55826                Not found
896178               RUIZ LARRINAGA N875000                99141533             Not found
99012114             LOPO RUANO N640000                    N968000              Not found
1111649              VINASPRE GARCIA l N103159             1124286              Not found
935725               ANDRES MARTINEZ JUAN JACINTO N133209  N640001              Not found
280364               GARCIA PARRO N173253                  N640000              Not found
P99090486            UGARTE ENRIQUE N312000                N156000              Not found
1547802              ONEDERRA N111637                      375000               Not found
472388               LUQUE OSUNA N328000                   N031000              Not found
53144                GUERRERO RAFAEL N156000               N328000              Not found
99109164             PEREZ FERNANDEZ DE LANDA N122054      N890000              Not found
99109167             ASPE_RODRIGO_ESTHER__99109167         N8590001             Not found
3201903              BARRIO CUENDE N437000                 1549266              Not found
1519562              MONTOYA ALFONSO N781000               99106030             Not found






======================================================


'''
