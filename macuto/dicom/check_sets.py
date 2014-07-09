

wd = os.path.expanduser('~/Dropbox/Documents/projects/santiago/scripts')
os.chdir(wd)

from itertools import combinations
def treefall(iterable):
    """

    :param iterable:
    :return:
    """
    num_elems = len(iterable)
    for i in range(num_elems, -1, -1):
        for c in combinations(iterable, i):
            yield c

from checklist import chklst
subjcontents = chklst.copy()

#modalities = ['dti', 'fm', 'flair', 't1']
modalities = ['dti', 'flair', 't1']

for modset in treefall(modalities):
    if not modset:
        continue

    print('\n')
    print("Who doesn't have {0}".format(' and '.join(modset)))
    mods = [m.strip() for m in modset]
    subjids = subjcontents.keys()

    for subj in subjids:
        contents = [c.strip() for c in subjcontents[subj].split(',')]
        sum_matches = sum([contents.count(m) for m in mods])
        if sum_matches == 0:
            print(subj)
            subjcontents.pop(subj)

#another try
import os
from glob import glob
from collections import OrderedDict

from itertools import combinations
def treefall(iterable):
    """

    :param iterable:
    :return:
    """
    num_elems = len(iterable)
    for i in range(num_elems, -1, -1):
        for c in combinations(iterable, i):
            yield c


modalities = ['diff', 'flair', 'anat']

#datadir = os.path.expanduser('~/Data/santiago/niftis')
datadir =  os.path.expanduser('/data/santiago/niftis')
subj_folders = os.listdir(datadir)

#fill the modalities dict
subj_modalities = OrderedDict()
for folder in subj_folders:
    mods = []
    for mod in modalities:
        if glob(os.path.join(datadir, folder, mod + '*')):
            mods.append(mod)

    subj_modalities[folder] = mods

#
subjcontents = subj_modalities.copy()
for modset in treefall(modalities):
    if not modset:
        continue

    print('\n')
    print("Who doesn't have {0}".format(' and '.join(modset)))
    mods = [m.strip() for m in modset]
    subjids = subjcontents.keys()

    for subj in subjids:
        contents = [c.strip() for c in subjcontents[subj]]
        sum_matches = sum([contents.count(m) for m in mods])
        if sum_matches == 0:
            print(subj)
            subjcontents.pop(subj)


csvids = {'921735': '148',
 'N100712': '108',
 '1560237': '111',
 '1365535': '147',
 '15817': '144',
 'N122054': '101',
 '925987': '315',
 '1578187': '152',
 '470663': '149',
 'N110626': '124',
 '1084809': '150',
 'N328000': '130',
 '248966': '245',
 '99117888': '311',
 'N281000': '228',
 '400742': '308',
 'N140000': '216',
 '99135285': '206',
 '55826': '120',
 '1952854': '322',
 '261676': '137',
 '685906': '323',
 '99106030': '103',
 '458039': '338',
 '1124286': '208',
 '1539140': '334',
 'N092449': '129',
 '1756647': '232',
 '1983029': '240',
 '1335553': '340',
 '1738856': '249',
 '99135289': '301',
 '1969022': '237',
 '99100167': '102',
 'N133209': '327',
 '707383': '321',
 'N859000': '115',
 'N031000': '217',
 '697987': '112',
 'N111637': '220',
 '1164737': '251',
 'P99143948': '121',
 '241599': '233',
 '99117812': '314',
 'N843000': '310',
 '99130019': '207',
 '886153': '213',
 'N115611': '132',
 'P99106025': '106',
 'N103159': '235',
 '661449': '242',
 'N437000': '241',
 '99109170': '306',
 'N104950': '201',
 '1939978': '246',
 'N718000': '302',
 '1954086': '227',
 '482470': '318',
 '1364867': '339',
 'N265000': '209',
 '1770861': '254',
 '890405': '140',
 '1328215': '143',
 'N640000': '116',
 'P99117222': '109',
 'N890000': '118',
 '2233682': '330',
 '1542604': '122',
 '99135286': '113',
 '1314109': '253',
 'N968000': '117',
 '470848': '151',
 'N750000': '222',
 'N8590001': '221',
 '671898': '252',
 '320712': '204',
 '99109169': '105',
 '19797': '332',
 '2136656': '236',
 'N173253': '326',
 'N406000': '223',
 '1956804': '331',
 '66334': '325',
 'N156000': '320',
 '1942888': '229',
 '45706': '304',
 '381249': '203',
 'N640001': '110',
 'P99145133': '211',
 '1325097': '123',
 '472315': '125',
 '683479': '244',
 '99106027': '107',
 '734615': '139',
 '26513': '138',
 '1740968': '210',
 '238394': '214',
 'N093000': '309',
 '1313224': '234',
 '35207': '248',
 '1328924': '231',
 '1554723': '328',
 'N102326': '215',
 '49612': '136',
 '936976': '333',
 'N875000': '224',
 '99119960': '317',
 '1113611': '135',
 'N718001': '225',
 '99130021': '205',
 'N781000': '219',
 '99072462': '239',
 '472717': '146',
 '375000': '218',
 '886783': '104',
 '46689': '307',
 '24525': '243',
 '687685': '142',
 '99105995': '305',
 '96049': '336',
 '1547734': '247',
 'N312000': '303',
 '1549266': '238',
 'N084248': '128',}


dicomids = [['55826', '455826'],
['46689'],
['1325097'],
['96049'],
['P99145133', '99145133'],
['1364867'],
['N133209', '935725'],
['734615'],
['2233682'],
['1954086'],
['886153'],
['1539140'],
['1542604'],
['1560237'],
['45706'],
['238394'],
['99100167'],
['1313224'],
['3201903', 'N437000'],
['921735'],
['P99117222', '99117222'],
['1740968'],
['261676'],
['1328215'],
['P99143948'],
['1770861'],
['482470', '1995804'],
['1942888'],
['1549266'],
['661449'],
['1365535'],
['400742'],
['35207'],
['99109170', '1278474'],
['99135285', '1092939'],
['381249'],
['683479'],
['19797'],
['99109169'],
['P99105995', '99105995'],
['99117812'],
['1341577', 'N843000'],
['P99148727', 'N084248'],
['99130019'],
['1328924'],
['890405'],
['24525'],
['39712', 'N110626'],
['1312383', 'N140000'],
['99074553', '1561125', 'N093000'],
['280364', 'N173253'],
['1983029'],
['1314109'],
['1939978'],
['1756647'],
['238542', 'N406000'],
['99135286'],
['470663'],
['53144', 'N156000'],
['99106027'],
['1547734'],
['671898'],
['2136656'],
['22403', 'N031000'],
['P99145371', 'N265000'],
['99130021'],
['26513'],
['1084809'],
['1543368', 'N859000'],
['1327760', 'N640001'],
['470848'],
['241599'],
['99012114', 'N640000'],
['472388', 'N328000'],
['687685'],
['99117888'],
['1522450', '375000'],
['49612'],
['925987'],
['1519562', 'N781000'],
['697987'],
['1578187'],
['1164737'],
['685906'],
['99106030'],
['15817'],
['444802', 'N750000'],
['248966'],
['1517716', 'N8590001'],
['1547802', 'N111637'],
['1113611'],
['99109164', 'N122054'],
['P99117223', 'P99106025'],
['P292762', 'N092449'],
['1969022'],
['1738856'],
['66334'],
['99143081', 'N718000'],
['1965124', 'N100712'],
['1738381', 'N102326'],
['896178', 'N875000', 'P99141533', '99141533'],
['P99148726', 'N718001'],
['1548357', 'N115611'],
['936976'],
['1952854'],
['1511222', '320712'],
['1554723'],
['1956804'],
['458039'],
['1124286'],
['1335553'],
['707383'],
['P99090486', 'N312000'],
['1529071', 'N104950'],
['P312430', 'N968000'],
['229418', 'N890000'],
['886783'],
['99135289', 'P99135289'],
['1574185', '99119960'],
['472717'],
['10102873', 'N281000'],
['1111649', 'N103159'],
['99072462'],
['472315']]


#subj_folders = ['10102873',
#'1084809',
#'1113611',
#'1124286',
#'1164737',
#'1312383',
#'1313224',
#'1314109',
#'1325097',
#'1327760',
#'1328215',
#'1328924',
#'1335553',
#'1364867',
#'1365535',
#'1517716',
#'1519562',
#'1522450',
#'1529071',
#'1539140',
#'1542604',
#'1543368',
#'1547734',
#'1548357',
#'1549266',
#'1554723',
#'1560237',
#'1578187',
#'15817',
#'1738381',
#'1738856',
#'1740968',
#'1756647',
#'1770861',
#'1939978',
#'1942888',
#'1952854',
#'1954086',
#'1956804',
#'1965124',
#'1969022',
#'19797',
#'1983029',
#'2136656',
#'2233682',
#'22403',
#'229418',
#'238394',
#'238542',
#'241599',
#'24525',
#'248966',
#'261676',
#'26513',
#'280364',
#'3201903',
#'320712',
#'35207',
#'375000',
#'381249',
#'39712',
#'400742',
#'444802',
#'455826',
#'45706',
#'458039',
#'46689',
#'470663',
#'470848',
#'472315',
#'472388',
#'472717',
#'482470',
#'49612',
#'53144',
#'55826',
#'661449',
#'66334',
#'671898',
#'683479',
#'685906',
#'687685',
#'697987',
#'707383',
#'734615',
#'886153',
#'886783',
#'890405',
#'896178',
#'921735',
#'925987',
#'935725',
#'936976',
#'96049',
#'99012114',
#'99100167',
#'99105995',
#'99106027',
#'99106030',
#'99109164',
#'99109169',
#'99109170',
#'99117812',
#'99117888',
#'99119960',
#'99130019',
#'99130021',
#'99135285',
#'99135286',
#'99135289',
#'99143081',
#'N031000',
#'N084248',
#'N092449',
#'N093000',
#'N100712',
#'N102326',
#'N103159',
#'N104950',
#'N110626',
#'N111637',
#'N115611',
#'N122054',
#'N133209',
#'N140000',
#'N156000',
#'N173253',
#'N265000',
#'N281000',
#'N312000',
#'N328000',
#'N406000',
#'N437000',
#'N640000',
#'N718000',
#'N718001',
#'N750000',
#'N781000',
#'N843000',
#'N859000',
#'N8590001',
#'N875000',
#'N890000',
#'N968000',
#'P292762',
#'P312430',
#'P99090486',
#'P99106025',
#'P99117222',
#'P99141533',
#'P99143948',
#'P99145133',
#'P99145371',
#'P99148726',]

superids = OrderedDict()
for id_set in dicomids:
    found = False
    for id in id_set:
        if id in csvids:
            found = True
            if csvids[id] in superids:
                print('Already put key {0}: {1}'.format(csvids[id], superids[csvids[id]]))
            else:
                superids[csvids[id]] = id_set
    if not found:
        print('Could not find key for {0}'.format(id_set))


def get_key_in_superids_values(superids, value):
    for k in superids:
        if value in superids[k]:
            return k

    return None

folder_moves = OrderedDict()
for folder in subj_folders:
    ariaid = get_key_in_superids_values(superids, folder)
    if ariaid is None:
        print('AriaID for {0} not found.'.format(folder))
    folder_moves[folder] = ariaid


for f in folder_moves:
    print('mv {0} {1}'.format(f, folder_moves[f]))
