import os
from loguru import logger

def setup_logger(worker_id=None):

    logger.remove()

    log_file = os.environ.get("LOG_FILE_PATH", "app.log")
    logger.add(log_file, level="INFO", rotation="10 MB")

    return logger


logger = setup_logger()
