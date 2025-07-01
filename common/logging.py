"""
Shared logging configuration for NovaOpsRemote
"""
import logging
import structlog
from .config import config


def setup_logging():
    """Configure structured logging for the application"""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'novaops_{config.NODE_ID}.log'),
            logging.StreamHandler()
        ]
    )


def get_logger(name: str = None):
    """Get a structured logger instance"""
    return structlog.get_logger(name)