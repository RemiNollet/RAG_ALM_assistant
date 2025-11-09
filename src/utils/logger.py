"""
Structured logger utility for RAG ALM Assistant.
"""

import logging
import sys
import os
import json
from datetime import datetime
from typing import Any, Dict


class JsonFormatter(logging.Formatter):
    """Format logs as JSON lines for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


def get_logger(name: str = "rag_alm_assistant") -> logging.Logger:
    """
    Return a configured logger with both console and JSON handler (optional).
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv("LOG_FORMAT", "human")  # or "json"

    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(log_level)
    handler = logging.StreamHandler(sys.stdout)

    if log_format == "json":
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger