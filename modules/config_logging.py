import logging


def config_logging(level: str) -> None:
    """
    'd' - DEBUG\n
    'i' - INFO\n
    'w' - WARNING\n
    'e' - ERROR\n
    'c' - CRITICAL
    """

    levels = {
        'd': logging.DEBUG,
        'i': logging.INFO,
        'w': logging.WARNING,
        'e': logging.ERROR,
        'c': logging.CRITICAL,
    }

    if level not in list(levels.keys()):
        raise Exception(f'level not in {list(levels.keys())}')

    format_ = '%(asctime)s-%(msecs)03d: %(levelname)s: %(message)s'
    datefmt = '%d-%m-%y %H-%M-%S'
    level_ = levels[level]
    logging.basicConfig(format=format_, datefmt=datefmt, level=level_)
