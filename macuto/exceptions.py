
import logging

log = logging.getLogger(__name__)


class LoggedError(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)
        log.error(message)