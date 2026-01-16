from __future__ import annotations

import logging
import os
import sys


def configure_logging() -> None:
    """
    Configura logging padrão do Python.

    Env:
      - LOG_LEVEL (default: INFO)
    """
    level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)

    logging.basicConfig(
        level=level,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Silencia alguns logs muito verbosos, se necessário
    logging.getLogger("httpx").setLevel(max(level, logging.WARNING))
    logging.getLogger("urllib3").setLevel(max(level, logging.WARNING))
