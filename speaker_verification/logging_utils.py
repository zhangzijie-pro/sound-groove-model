import logging
import os
from datetime import datetime


def setup_logger(out_dir: str, logger_name: str = "train_logger"):
    os.makedirs(out_dir, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    log_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(out_dir, f"train_{log_time}.log")

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info("Logger initialized. Log file: %s", log_file)
    return logger, log_file