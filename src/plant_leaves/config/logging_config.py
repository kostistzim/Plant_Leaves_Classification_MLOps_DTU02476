from loguru import logger


def configure_logger():
    logger.configure(extra={"prefix": "DEFAULT"})
    logger.remove(0)
    logger.add("logs/logging.log", format="[{extra[prefix]}] | {time:MMMM D, YYYY > HH:mm:ss} | {level} | {message}")


configure_logger()

__all__ = ["logger"]
