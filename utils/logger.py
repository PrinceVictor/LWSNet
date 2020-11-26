import logging
import os
import sys
import time

def setup_logger(name, save_path=None):

    file_name = time.strftime('-%Y-%m-%d-%H-%M', time.localtime(time.time()))
    name = name.replace(".", "_")

    if os.path.dirname(name) == "":
        file_name = name + file_name + ".log"
    else:
        file_name = os.path.dirname(name) + file_name + ".log"

    log_formatter = logging.Formatter(
        "[%(asctime)s %(filename)s:%(lineno)s] %(levelname)s: %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S')

    logger = logging.getLogger(file_name)

    stream_handler = logging.StreamHandler(stream=sys.stderr)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)

    if file_name in [h.name for h in logger.handlers]:
        return

    if save_path is not None:
        if os.path.dirname(save_path) != '':
            if not os.path.isdir(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))

        file_handler = logging.FileHandler(os.path.join(save_path, file_name))
        file_handler.set_name(file_name)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)

    return logger


