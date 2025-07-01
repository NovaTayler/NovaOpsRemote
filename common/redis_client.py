"""
Redis client for caching, pub/sub, and node pings
"""
import redis.asyncio as redis
from .config import config
from .logging import get_logger

logger = get_logger(__name__)


def get_redis_client():
    """Get Redis client instance"""
    return redis.Redis.from_url(config.REDIS_URL, decode_responses=True)


# Global Redis client instance
redis_client = get_redis_client()