import multiprocessing

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = 1
worker_class = 'sync'
worker_connections = 1000
timeout = 120
keepalive = 2

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Process naming
proc_name = 'gunicorn_fluency_predictor'

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Memory optimization
max_requests = 1000
max_requests_jitter = 50
preload_app = True

# Limit worker memory usage
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190 