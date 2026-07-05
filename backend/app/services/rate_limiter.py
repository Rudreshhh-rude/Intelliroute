import time
import logging
from threading import Lock
from typing import Dict, Tuple

logger = logging.getLogger("intelliroute.rate_limiter")

class TokenBucket:
    def __init__(self, capacity: float = 60.0, refill_rate: float = 1.0):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_updated = time.time()
        self.lock = Lock()

    def consume(self) -> Tuple[bool, float]:
        """Consumes a token from the bucket.
        
        Returns (is_allowed, remaining_tokens).
        """
        with self.lock:
            now = time.time()
            elapsed = now - self.last_updated
            self.last_updated = now
            
            # Refill tokens based on elapsed time
            self.tokens = min(self.capacity, self.tokens + (elapsed * self.refill_rate))
            
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True, self.tokens
            return False, self.tokens


class RateLimiter:
    def __init__(self, capacity: float = 60.0, refill_rate: float = 1.0):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.buckets: Dict[str, TokenBucket] = {}
        self.lock = Lock()

    def is_allowed(self, api_key: str) -> Tuple[bool, float]:
        """Checks if a request from the given API key is allowed.
        
        Returns (is_allowed, remaining_tokens).
        """
        with self.lock:
            if api_key not in self.buckets:
                self.buckets[api_key] = TokenBucket(
                    capacity=self.capacity,
                    refill_rate=self.refill_rate
                )
            bucket = self.buckets[api_key]
            
        return bucket.consume()
