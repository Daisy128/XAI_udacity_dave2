import logging
import csv
from dataclasses import asdict
import json
import os
from typing import List
import sys
import numpy as np

from ..Simulator.Scenario import ScenarioOutcome, OfflineScenarioOutcome
from .custom_types import LOGGING_LEVEL
from PIL import Image
import gc


class CSVLogHandler(logging.FileHandler):
    """
    Util class to log perturbation output and metrics

    :param: filename="logs.csv": String name of log file
    :param: mode="w": Mode of the logger. Here we can use options such as "w", "a", ...
    :param: encoding=None: Encoding of the file
    """

    def __init__(self, filename="logs.csv", mode="w", encoding=None):
        super().__init__(filename, mode, encoding, delay=False)
        self.writer = csv.writer(
            self.stream, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        self.current_row = []

    def emit(self, record):
        if isinstance(record.msg, (list, tuple)):
            self.current_row.extend(record.msg)
        else:
            self.current_row.append(record.msg)

    def flush_row(self):
        if self.current_row:
            self.writer.writerow(self.current_row)
            self.flush()
            self.current_row = []

class GlobalLog:
    """This class is used to log across different models in the project"""

    def __init__(self, logger_prefix: str, log_file: str=None):
        """
        We use the logger_prefix to distinguish between different loggers

        :param logger_prefix: prefix of the logger
        """
        self.logger = logging.getLogger(logger_prefix)
        # avoid creating another logger if it already exists
        if len(self.logger.handlers) == 0:
            self.logger = logging.getLogger(logger_prefix)
            self.logger.setLevel(level=LOGGING_LEVEL)

            formatter = logging.Formatter("%(levelname)s: %(name)s: %(message)s")
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(formatter)
            ch.setLevel(level=logging.DEBUG)
            self.logger.addHandler(ch)

            if log_file:
                fh = logging.FileHandler(log_file)
                fh.setFormatter(formatter)
                fh.setLevel(level=LOGGING_LEVEL)
                self.logger.addHandler(fh)

    def debug(self, message):
        """Log debug message"""
        self.logger.debug(message)

    def info(self, message):
        """Log info message"""
        self.logger.info(message)

    def warn(self, message):
        """Log warning message"""
        self.logger.warn(message)

    def error(self, message):
        """Log error message"""
        self.logger.error(message)

    def critical(self, message):
        """Log critical message"""
        self.logger.critical(message)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
