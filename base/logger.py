from pathlib import Path
from base.config import config as conf
import logging
import os
import colorlog

log_dia_path = Path(__file__).parents[1] / "log" / conf.logger.log_file

def setup_logging(log_file=log_dia_path):
    os.makedirs(os.path.dirname(log_file),exist_ok=True)
    logger = logging.getLogger("RAG")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file,encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = colorlog.ColoredFormatter("%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger

logger = setup_logging()













