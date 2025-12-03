"""
Celery Application Configuration
================================

Celery worker for asynchronous task processing and scheduled jobs.

Features:
- Background task processing
- Scheduled tasks (celery beat)
- Task monitoring with Flower
- Redis as message broker
- PostgreSQL for result backend

Author: Magnus Trading Platform
Created: 2025-11-21
"""

import os
from celery import Celery
from celery.schedules import crontab
from kombu import Exchange, Queue

# Initialize Celery app
app = Celery('magnus')

# Configuration
app.conf.update(
    # Broker settings
    broker_url=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    result_backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0'),

    # Serialization
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,

    # Task routing
    task_routes={
        'src.services.tasks.sync_kalshi_markets': {'queue': 'market_data'},
        'src.services.tasks.update_stock_prices': {'queue': 'market_data'},
        'src.services.tasks.generate_predictions': {'queue': 'predictions'},
        'src.services.tasks.send_alerts': {'queue': 'notifications'},
        'src.services.tasks.cleanup_old_data': {'queue': 'maintenance'},
    },

    # Queue definitions
    task_queues=(
        Queue('default', Exchange('default'), routing_key='default'),
        Queue('market_data', Exchange('market_data'), routing_key='market_data'),
        Queue('predictions', Exchange('predictions'), routing_key='predictions'),
        Queue('notifications', Exchange('notifications'), routing_key='notifications'),
        Queue('maintenance', Exchange('maintenance'), routing_key='maintenance'),
    ),

    # Task execution
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=4,
    worker_max_tasks_per_child=1000,

    # Result backend
    result_expires=3600,  # 1 hour
    result_backend_transport_options={'master_name': 'mymaster'},

    # Beat schedule (scheduled tasks)
    beat_schedule={
        # Sync Kalshi markets every 5 minutes
        'sync-kalshi-markets': {
            'task': 'src.services.tasks.sync_kalshi_markets',
            'schedule': crontab(minute='*/5'),
            'options': {'queue': 'market_data'}
        },

        # Update stock prices every 1 minute during market hours
        'update-stock-prices': {
            'task': 'src.services.tasks.update_stock_prices',
            'schedule': crontab(minute='*/1', hour='9-16', day_of_week='mon-fri'),
            'options': {'queue': 'market_data'}
        },

        # Generate AI predictions every 15 minutes
        'generate-predictions': {
            'task': 'src.services.tasks.generate_predictions',
            'schedule': crontab(minute='*/15'),
            'options': {'queue': 'predictions'}
        },

        # Send alerts every hour
        'send-hourly-alerts': {
            'task': 'src.services.tasks.send_alerts',
            'schedule': crontab(minute=0),
            'options': {'queue': 'notifications'}
        },

        # Cleanup old data daily at 2 AM
        'cleanup-old-data': {
            'task': 'src.services.tasks.cleanup_old_data',
            'schedule': crontab(hour=2, minute=0),
            'options': {'queue': 'maintenance'}
        },

        # Sync Discord messages every 5 minutes (with premium alert notifications)
        'sync-discord-messages': {
            'task': 'src.services.tasks.sync_discord_messages',
            'schedule': crontab(minute='*/5'),
            'options': {'queue': 'market_data'}
        },

        # Update earnings calendar daily at 6 AM
        'update-earnings-calendar': {
            'task': 'src.services.tasks.update_earnings_calendar',
            'schedule': crontab(hour=6, minute=0),
            'options': {'queue': 'market_data'}
        },

        # Warm caches every 30 minutes
        'warm-caches': {
            'task': 'src.services.tasks.warm_caches',
            'schedule': crontab(minute='*/30'),
            'options': {'queue': 'maintenance'}
        },

        # Sync XTrades messages to RAG daily at 1 AM
        'sync-xtrades-to-rag': {
            'task': 'src.services.tasks.sync_xtrades_to_rag',
            'schedule': crontab(hour=1, minute=0),
            'options': {'queue': 'maintenance'}
        },

        # Sync Discord messages to RAG daily at 2 AM
        'sync-discord-to-rag': {
            'task': 'src.services.tasks.sync_discord_to_rag',
            'schedule': crontab(hour=2, minute=0),
            'options': {'queue': 'maintenance'}
        },
    },
)

# Auto-discover tasks from installed apps
app.autodiscover_tasks([
    'src.services',
])


# Task decorator shortcuts
@app.task(bind=True, name='test_celery')
def test_celery(self) -> None:
    """Test task to verify Celery is working"""
    return f"Celery is working! Task ID: {self.request.id}"


if __name__ == '__main__':
    app.start()
