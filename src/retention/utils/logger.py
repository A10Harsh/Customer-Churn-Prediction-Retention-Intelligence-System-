"""Centralized structured logging utilities."""

import json
import logging
import os
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    """Format log records as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(
                record.created,
                tz=UTC,
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload)


def get_logger(name: str) -> logging.Logger:
    """Return a configured application logger."""

    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)

    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    logger.propagate = False

    return logger
