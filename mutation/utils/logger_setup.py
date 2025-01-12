import logging
import os
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

LOGS_ROOT = os.path.join(PROJECT_ROOT, 'logs')

if not os.path.exists(LOGS_ROOT):
    os.mkdir(LOGS_ROOT)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter.datefmt = '%m/%d/%Y %I:%M:%S %p'

# create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)

# create app file handler
app_log_path = os.path.join(LOGS_ROOT, 'app_log.txt')
fha = logging.FileHandler(app_log_path)
fha.setLevel(logging.DEBUG)
fha.setFormatter(formatter)

# create user file handler
user_log_path = os.path.join(LOGS_ROOT, 'user_log.txt')
fhu = logging.FileHandler(user_log_path)
fhu.setLevel(logging.DEBUG)
fhu.setFormatter(formatter)


def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    logger.addHandler(ch)
    logger.addHandler(fha)
    logger.addHandler(fhu)

    return logger

