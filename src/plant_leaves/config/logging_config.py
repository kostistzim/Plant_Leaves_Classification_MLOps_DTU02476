from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def configure_logger():
    logger.configure(extra={"prefix": "DEFAULT"})
    # try:
    #     logger.remove(0)
    # except ValueError as error:
    #     logger.warning(error)
    #     pass
    logger.remove()
    logger.add(
        f"{PROJECT_ROOT}/logs/logging.log",
        format="[{extra[prefix]}] | {time:MMMM D, YYYY > HH:mm:ss} | {level} | {message}",
    )


configure_logger()

__all__ = ["logger"]
