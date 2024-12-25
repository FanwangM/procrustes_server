import multiprocessing

# Number of worker processes
workers = multiprocessing.cpu_count() * 2 + 1

# Number of threads per worker
threads = 4

# Maximum number of pending connections
backlog = 2048

# Maximum number of requests a worker will process before restarting
max_requests = 1000
max_requests_jitter = 50

# Timeout for worker processes
timeout = 300

# Keep-alive timeout
keepalive = 5

# Log level
loglevel = 'info'

# Access log format
accesslog = '-'
errorlog = '-'

# Bind address
bind = '0.0.0.0:8000'

# Worker class
worker_class = 'gevent'
