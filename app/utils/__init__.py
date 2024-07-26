import logging
import os
import time
from functools import wraps


import structlog
from dotenv import load_dotenv
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_asyncio
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException


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


class QdrantClientRetry:
    def __init__(self, client, max_retries=3, retry_delay=1):
        self.client = client
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.log = structlog.get_logger()

    def retry_decorator(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(self.max_retries):
                try:
                    return func(*args, **kwargs)
                except ResponseHandlingException as e:
                    if "Server disconnected without sending a response" in str(e):
                        if attempt < self.max_retries - 1:
                            self.log.warning(
                                f"Qdrant HTTP attempt failed. Retrying in {self.retry_delay} seconds...",
                                attempt=attempt,
                            )
                            time.sleep(self.retry_delay)
                        else:
                            self.log.warning(
                                f"Max retries reached. Raising the exception.",
                                max_retries=self.max_retries,
                            )
                            raise
                    else:
                        raise

        return wrapper

    def __getattr__(self, name):
        attr = getattr(self.client, name)
        if callable(attr):
            return self.retry_decorator(attr)
        return attr


def setup_qdrant_client():
    """Setup Qdrant client"""
    client = QdrantClient(
        os.getenv("QDRANT_API_URL"),
        port=os.getenv("QDRANT_API_PORT"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    return QdrantClientRetry(client, max_retries=5, retry_delay=2)
