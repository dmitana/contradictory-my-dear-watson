import logging
import sys


def initialize_logger(name: str, logging_level: int) -> None:
    """
    Initialize logger `name`.

    :param name: name of logger to be initialized.
    :param logging_level: log level.
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging_level)

    # Create console handler
    ch = logging.StreamHandler(stream=sys.stderr)
    ch.setLevel(logging_level)

    # Create formatter and add it to the handlers
    fmt = '%(asctime)-15s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S')
    ch.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(ch)
