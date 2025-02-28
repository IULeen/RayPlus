# 存储系统对接模块

# 0. Process 存活进程，不在该模块中管理
# 1. Ray Object Store，Ray 官方提供的存储系统
# 目前支持的存储系统
# 2. Redis/AWS ElastiCache
# 3. MinIO/AWS S3

from enum import Enum
from typing import Union, Optional

class StorageType(Enum):
    REDIS = "redis"
    MINIO = "minio"

class StorageConfig:
    def __init__(
        self,
        storage_type: Union[StorageType, str],
        host: Optional[str] = None,
        port: Optional[int] = None,
        **kwargs
    ):
        if isinstance(storage_type, str):
            storage_type = StorageType(storage_type.lower())
        self.storage_type = storage_type
        self.host = host
        self.port = port
        self.extra_config = kwargs

def get_storage(config: StorageConfig):
    """
    根据配置创建对应的存储客户端
    
    Args:
        config: StorageConfig 配置对象
    
    Returns:
        存储客户端实例
    """
    if config.storage_type ==  StorageType.REDIS:
        from .redis_store import RedisStorage
        return RedisStorage(config.host, config.port, **config.extra_config)
    elif config.storage_type == StorageType.MINIO:
        from .minio_store import MinioStorage
        return MinioStorage(config.host, config.port, **config.extra_config)
    else:
        raise ValueError(f"不支持的存储类型: {config.storage_type}")
