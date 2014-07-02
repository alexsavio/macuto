
import logging

log = logging.getLogger(__name__)


class LoggedError(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)
        log.error(message)


class LoggedValueError(LoggedError, ValueError):
    def __init__(self, message):
        ValueError.__init__(self, message)
        LoggedError.__init__(self, message)


class FolderNotFound(LoggedError):
    pass


class FolderAlreadyExists(LoggedError):
    pass

