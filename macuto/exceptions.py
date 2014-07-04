
import logging

log = logging.getLogger(__name__)


class LoggedError(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)
        log.error(message)


class PathNotFoundError(LoggedError):
    path_type = 'path'

    def __init__(self, file_path, message=None):

        msg = 'Could not find {0} {1}.'.format(self.path_type,
                                               file_path)
        if message is not None:
            msg += '. ' + message

        Exception.__init__(self, msg)
        log.error(msg)


class FileNotFound(PathNotFoundError):
    path_type = 'file'


class FolderNotFound(PathNotFoundError):
    path_type = 'folder'


class PathAlreadyExists(LoggedError):
    path_type = 'path'

    def __init__(self, file_path, message=None):

        msg = '{0} {1} already exists {1}.'.format(self.path_type.capitalize(),
                                                   file_path)
        if message is not None:
            msg += '. ' + message

        Exception.__init__(self, msg)
        log.error(msg)


class FileAlreadyExists(LoggedError):
    path_type = 'file'


class FolderAlreadyExists(LoggedError):
    path_type = 'folder'
