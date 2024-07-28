import logging

import structlog
from dotenv import load_dotenv
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_asyncio


load_dotenv()


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def configure_structlog():
    """Configure structlog and logging"""

    structlog.configure(
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(
        format="%(message)s", level=logging.INFO, handlers=[TqdmLoggingHandler()]
    )
    # ignore INFO messages from httpx / qdrant
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("shap").setLevel(logging.WARNING)
