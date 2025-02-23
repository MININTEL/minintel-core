"""
Machine Intelligence Node - Advanced Logging System

Enhances structured logging with async support, log rotation, 
and JSON-based structured output for AI training and monitoring.

Author: Machine Intelligence Node Development Team
"""

import logging
import os
import sys
import json
import asyncio
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional

class AsyncLogHandler(logging.Handler):
    """Asynchronous log handler for improved performance in AI workloads."""
    def emit(self, record):
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, self._write_log, record)

    def _write_log(self, record):
        msg = self.format(record)
        sys.stdout.write(msg + "\n")

class JSONFormatter(logging.Formatter):
    """Custom formatter to log output as structured JSON for monitoring."""
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "filename": record.filename,
            "line": record.lineno
        }
        return json.dumps(log_entry)

class Logger:
    """
    A configurable logging system with support for async logging,
    log rotation, structured JSON output, and console color coding.
    """
    def __init__(self, log_file: Optional[str] = None, log_level: str = "INFO", json_output: bool = False):
        """
        Initializes the logger.

        Args:
            log_file (str, optional): Path to save log file (default: None).
            log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            json_output (bool): Whether to enable structured JSON logging.
        """
        self.log_file = log_file
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.json_output = json_output

        # Create logger instance
        self.logger = logging.getLogger("MachineIntelligenceNode")
        self.logger.setLevel(self.log_level)

        # Console handler
        console_handler = AsyncLogHandler()
        console_handler.setFormatter(self._get_formatter())
        self.logger.addHandler(console_handler)

        # File handler with rotation
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
            file_handler.setFormatter(self._get_formatter(file_output=True))
            self.logger.addHandler(file_handler)

    def _get_formatter(self, file_output: bool = False) -> logging.Formatter:
        """
        Returns a formatter with color support for console or JSON formatting.

        Args:
            file_output (bool): Whether the log is for file output.

        Returns:
            logging.Formatter: Configured log formatter.
        """
        if self.json_output:
            return JSONFormatter()

        reset = "\033[0m"
        colors = {
            "DEBUG": "\033[94m",     # Blue
            "INFO": "\033[92m",      # Green
            "WARNING": "\033[93m",   # Yellow
            "ERROR": "\033[91m",     # Red
            "CRITICAL": "\033[95m",  # Purple
        }

        log_format = "%(asctime)s | %(levelname)s | %(message)s"
        if not file_output:
            log_format = log_format.replace(
                "%(levelname)s", f"{colors.get('%(levelname)s', '')}%(levelname)s{reset}"
            )

        return logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    def debug(self, message: str):
        """Logs a DEBUG-level message."""
        self.logger.debug(message)

    def info(self, message: str):
        """Logs an INFO-level message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Logs a WARNING-level message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Logs an ERROR-level message."""
        self.logger.error(message)

    def critical(self, message: str):
        """Logs a CRITICAL-level message."""
        self.logger.critical(message)

# Example Usage
if __name__ == "__main__":
    log_file = "logs/system.log"
    logger = Logger(log_file=log_file, log_level="DEBUG", json_output=False)

    logger.debug("Debugging AI pipeline.")
    logger.info("Machine Intelligence Node started successfully.")
    logger.warning("Potential issue detected in dataset processing.")
    logger.error("Failed to load model checkpoint.")
    logger.critical("System encountered a critical failure.")
