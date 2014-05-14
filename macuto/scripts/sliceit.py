__author__ = 'alexandre'

import baker

#logging config
logging.basicConfig(level=logging.DEBUG, filename='sliceit.log',
                    format="%(asctime)-15s %(message)s")
log = logging.getLogger('sliceit')

@baker.command(default=True,
               shortopts={'output_dir': 'o',
                          'file_regex': 'i'})
def one(output_dir, file_list1, file_list2, dpi=150, **kwargs):





@baker.command(default=True,
               shortopts={'subjfolder': 'i',
                          'idregex': 'r',
                          'not_rename_folder': 'f'})
def subject(subjfolder, idregex='', not_rename_folder=False):
    """Anonymizes the entire subject folder.
    First renames the main folder to the acquisition ID, then
    it looks inside for subdirectories with DICOM files and
    anonymizes each of them.

    :param subjfolder: Path to the subject folder

    :param idregex: Regex to search for ID in folder name

    :param not_rename_folder: If this flag is set, will not rename
                              subjects' folder
    """

if __name__ == '__main__':
    baker.run()