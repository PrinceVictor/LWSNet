import logging
import os
import sys


def setup_logger(name, save_path=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.DEBUG)

    log_formatter = logging.Formatter(
        "[%(asctime)s %(filename)s:%(lineno)s] %(levelname)s: %(message)s")

    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)

    if save_path is not None:
        file_handler = logging.FileHandler(os.path.join(save_path, name))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)

    return logger


