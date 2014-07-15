
import os
import sys
import yaml
import logging.config

from .config import LOG_LEVEL


def setup_logging(default_path=os.path.join(os.path.dirname(__file__),
                                            'logging.yaml'),
                  default_level=LOG_LEVEL,
                  env_key='MACUTO_LOG_CFG'):
    """Setup logging configuration

    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.load(f)
        logging.config.dictConfig(config)
        print('Started logging from config file {0}.'.format(path))
    else:
        logging.basicConfig(level=default_level)
        print('Started default logging.')