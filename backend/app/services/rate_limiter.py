import os
import time
import logging
from threading import Lock
from typing import Dict, Tuple

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger("intelliroute.rate_limiter")

class TokenBucket:
    def __init__(self, capacity: float = 60.0, refill_rate: float = 1.0):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_updated = time.time()
        self.lock = Lock()

    def consume(self) -> Tuple[bool, float]:
        with self.lock:
            now = time.time()
            elapsed = now - self.last_updated
            self.last_updated = now
            self.tokens = min(self.capacity, self.tokens + (elapsed * self.refill_rate))
            
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True, self.tokens
            return False, self.tokens

class RedisRateLimiter:
    def __init__(self, redis_client, capacity: float = 60.0, refill_rate: float = 1.0):
        self.redis = redis_client
        self.capacity = capacity
        self.refill_rate = refill_rate

    def is_allowed(self, api_key: str) -> Tuple[bool, float]:
        script = """
        local capacity = tonumber(ARGV[1])
        local refill_rate = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])
        
        local last_updated = tonumber(redis.call('HGET', KEYS[1], 'last_updated')) or now
        local tokens = tonumber(redis.call('HGET', KEYS[1], 'tokens')) or capacity
        
        local elapsed = math.max(0, now - last_updated)
        tokens = math.min(capacity, tokens + (elapsed * refill_rate))
        
        local allowed = 0
        if tokens >= 1.0 then
            tokens = tokens - 1.0
            allowed = 1
        end
        
        redis.call('HSET', KEYS[1], 'last_updated', now, 'tokens', tokens)
        redis.call('EXPIRE', KEYS[1], 3600)
        
        return {allowed, tostring(tokens)}
        """
        now = time.time()
        result = self.redis.eval(script, 1, f"rate_limit:{api_key}", self.capacity, self.refill_rate, now)
        allowed = result[0] == 1
        tokens = float(result[1])
        return allowed, tokens

class RateLimiter:
    def __init__(self, capacity: float = 60.0, refill_rate: float = 1.0):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.buckets: Dict[str, TokenBucket] = {}
        self.lock = Lock()
        
        redis_url = os.environ.get("REDIS_URL")
        self.redis_backend = None
        if redis_url and REDIS_AVAILABLE:
            try:
                r = redis.Redis.from_url(redis_url)
                r.ping()
                self.redis_backend = RedisRateLimiter(r, capacity, refill_rate)
                logger.info("Successfully connected to Redis. Using RedisRateLimiter.")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis at {redis_url}: {e}. Falling back to MemoryRateLimiter.")

    def is_allowed(self, api_key: str) -> Tuple[bool, float]:
        if self.redis_backend:
            try:
                return self.redis_backend.is_allowed(api_key)
            except Exception as e:
                logger.error(f"Redis rate limiting failed: {e}. Falling back to memory.")
        
        with self.lock:
            if api_key not in self.buckets:
                self.buckets[api_key] = TokenBucket(
                    capacity=self.capacity,
                    refill_rate=self.refill_rate
                )
            bucket = self.buckets[api_key]
            
        return bucket.consume()
