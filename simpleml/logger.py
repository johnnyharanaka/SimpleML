"""Centralized logger for SimpleML."""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_log = logging.getLogger("simpleml")

log_info = _log.info
log_warning = _log.warning
log_error = _log.error

def get_logger() -> logging.Logger:
    """Return the SimpleML logger."""
    return _log
