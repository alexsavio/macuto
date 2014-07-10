

class PathNotFoundError(Exception):
    path_type = 'path'

    def __init__(self, file_path, message=None):

        msg = 'Could not find {0} {1}.'.format(self.path_type,
                                               file_path)
        if message is not None:
            msg += '. ' + message

        Exception.__init__(self, msg)


class PathAlreadyExists(Exception):
    path_type = 'path'

    def __init__(self, file_path, message=None):

        msg = '{0} {1} already exists {1}.'.format(self.path_type.capitalize(),
                                                   file_path)
        if message is not None:
            msg += '. ' + message

        Exception.__init__(self, msg)


class FileNotFound(PathNotFoundError):
    path_type = 'file'


class FolderNotFound(PathNotFoundError):
    path_type = 'folder'


class FileAlreadyExists(PathAlreadyExists):
    path_type = 'file'


class FolderAlreadyExists(PathAlreadyExists):
    path_type = 'folder'
