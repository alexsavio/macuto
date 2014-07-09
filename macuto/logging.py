
import os
import yaml
import logging.config

from .config import LOG_LEVEL


def setup_logging(default_path='logging.yaml', default_level=LOG_LEVEL,
                  env_key='MACUTO_LOG_CFG'):
    """Setup logging configuration

    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'r') as f:
            config = yaml.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)