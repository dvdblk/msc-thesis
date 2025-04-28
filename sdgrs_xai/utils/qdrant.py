import time
import os
import structlog
from qdrant_client import QdrantClient
from typing import List, Dict, Any, Optional
from functools import wraps
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException


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


def setup_qdrant_client(
    url: Optional[str], port: Optional[int], api_key: Optional[str]
):
    """Set up and return a Qdrant client."""
    client = QdrantClient(
        url=url or os.getenv("QDRANT_API_URL"),
        port=port or os.getenv("QDRANT_API_PORT"),
        api_key=api_key or os.getenv("QDRANT_API_KEY"),
    )
    return QdrantClientRetry(client, max_retries=5, retry_delay=2)
