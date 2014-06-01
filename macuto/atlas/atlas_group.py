
import os
import logging
from .atlas import Atlas, LabelAtlas, StatsAtlas
from .atlas_files import AtlasFiles
from ..strings import search_list


log = logging.getLogger(__name__)


class AtlasGroup:
    """
    """

    def __init__(self):
        self.atlases = {}
        self.create()

    def create(self):
        """
        atlases is a string->img dict
        This function fills the atlases dict if empty with the data obtained
        from each .xml file found in FSL atlas paths.

        """
        atlas_files = AtlasFiles()

        if len(self.atlases) == 0:
            atlas_dirs = atlas_files.get_atlas_path_elements()

            for d in atlas_dirs:
                if os.path.exists(d):
                    files = search_list(os.listdir(d), '.xml$')
                    for f in files:
                        self.read_atlas(d, f)

    def read_atlas(self, path, file_name):
        """
        Adds to self.atlases the atlas in path/file_name

        Parameters
        ----------
        path: string
        Path where the atlas files are

        file_name: string
        Atlas file name
        """
        atlas_files = AtlasFiles()
        atlas = atlas_files.read_xml_atlas(path, file_name)
        self.atlases[atlas.name] = atlas

    def get_atlas_by_name(self, name):
        """
        Returns the atlas with the given name

        Parameters
        ----------
        name: string
        Atlas name

        """
        return self.atlases[name] if name in self.atlases else None


    def get_compatible_atlases(ref_img):
        """
        """
        for nom in atlases:
            atlases[nom].select_compatible_images(ref_img)