from celery import Celery

celery = Celery("procrustes_server", broker="redis://redis:6379/0", backend="redis://redis:6379/0")

celery.conf.update(
    worker_max_tasks_per_child=1000,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    broker_pool_limit=None,
    broker_connection_timeout=30,
    result_expires=3600,  # Results expire after 1 hour
    task_track_started=True,
    task_time_limit=300,  # 5 minutes
    task_soft_time_limit=240,  # 4 minutes
    worker_concurrency=4,  # Number of worker processes per Celery worker
)
