import logging
import sys
from typing import Dict

from torch.utils.tensorboard.writer import SummaryWriter


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


def tensorboard_add_scalars(
    writer: SummaryWriter,
    mode: str,
    scalars: Dict[str, float],
    global_step: int
) -> None:
    """
    Add `scalars` to TensorBoard `writer`.

    :param writer: TensorBoard writer.
    :param mode: train/test/... mode.
    :param scalars: dictionary of scalar name and value.
    :param global_step: global step value to record.
    """
    for key, value in scalars.items():
        writer.add_scalar(f'{key}/{mode}', value, global_step)
