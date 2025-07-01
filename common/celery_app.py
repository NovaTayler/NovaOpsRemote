"""
Celery instance configured with shared broker and backend
"""
from celery import Celery
from .config import config
from .logging import get_logger

logger = get_logger(__name__)


def create_celery_app():
    """Create and configure Celery application"""
    app = Celery('novaops', broker=config.CELERY_BROKER_URL, backend=config.CELERY_RESULT_BACKEND)
    
    # Configure Celery
    app.conf.update(
        task_reject_on_worker_lost=True,
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        task_serializer='json',
        result_serializer='json',
        accept_content=['json'],
        result_expires=3600,
    )
    
    return app


# Global Celery app instance
celery_app = create_celery_app()