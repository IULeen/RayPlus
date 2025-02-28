# config/settings.py

##############################
# Redis 初始化参数（根据需要调整）
##############################
REDIS_HOST = "192.168.1.180"
REDIS_PORT = 32737

##############################
# MINIO 初始化参数（根据需要调整）
##############################
MINIO_HOST = "127.0.0.1"
MINIO_PORT = 9000
MINIO_ACCESS_KEY = "admin"
MINIO_SECRET_KEY = "root123456"
MINIO_BUCKET_NAME = "actorless-bucket"

##############################
# Ray 初始化参数（根据需要调整）
##############################
# 使用 CPU 资源代替 Memory
# 1 CPU = 10MB Memory

RAY_INIT_CONFIG = {
    "num_cpus": 100,  # 可用的CPU数量
    "resouces": {"memory": 4000},  # 可用的Memory数量，增加到5000MB
    "_system_config": {
        "object_timeout_milliseconds": 200
    }
}

##############################
# 存储层级配置
##############################
# 定义不同存储层级的性能特性
STORAGE_LEVELS = {
    # 进程内存储 - 最快但不持久
    "PROCESS": {
        "read_latency": 0.0001,  # 读取延迟（秒）
        "write_latency": 0.0002,  # 写入延迟（秒）
        "cost_per_gb": 0.5,      # 每GB成本（相对值）
        "availability": 0.99     # 可用性（0-1）
    },
    # 内存存储 - 快速但有限
    "MEMORY": {
        "read_latency": 0.001,
        "write_latency": 0.002,
        "cost_per_gb": 0.2,
        "availability": 0.995
    },
    # Redis存储 - 中等速度，持久
    "REDIS": {
        "read_latency": 0.01,
        "write_latency": 0.02,
        "cost_per_gb": 0.1,
        "availability": 0.999
    },
    # MinIO存储 - 较慢但成本低
    "MINIO": {
        "read_latency": 0.1,
        "write_latency": 0.2,
        "cost_per_gb": 0.01,
        "availability": 0.9999
    }
}