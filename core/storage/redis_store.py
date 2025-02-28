from typing import Any, Optional, Dict, List
import redis
from redis.client import Redis
import pickle
import logging

log = logging.getLogger(__name__)

class RedisStorage:
    """Redis 存储管理类
    
    提供基于 Redis 的键值存储功能
    """
    
    def __init__(self, host: str, port: int, connection_pool=None):
        """初始化 Redis 连接
        
        Args:
            host: Redis 服务器地址
            port: Redis 服务器端口
            connection_pool: Redis 连接池，可选
        """
        if connection_pool:
            self.redis_client = redis.Redis(connection_pool=connection_pool)
        else:
            self.redis_client = redis.Redis(host=host, port=port)

    def set(self, key: str, value: bytes, expire: int = None) -> bool:
        """存储数据到Redis，带数据大小检查"""
        try:
            # 存储数据大小
            # TODO: 加上数据大小就可以检索到了，不知道为啥，待检查与解决
            data_size = len(value)
            pipeline = self.redis_client.pipeline()
            pipeline.set(key, value)
            pipeline.set(f"{key}:size", data_size)
            if expire:
                pipeline.expire(key, expire)
                pipeline.expire(f"{key}:size", expire)
            pipeline.execute()
            return True
        except Exception as e:
            log.error(f"Failed to store data in Redis: {e}")
            return False

    def get(self, key: str) -> Optional[bytes]:
        """从Redis获取数据，带完整性检查"""
        try:
            # 使用pipeline原子操作
            # TODO: 加上数据大小就可以检索到了，不知道为啥，待检查与解决
            pipeline = self.redis_client.pipeline()
            pipeline.get(key)
            pipeline.get(f"{key}:size")
            value, stored_size = pipeline.execute()

            if value is None or stored_size is None:
                return None

            # 检查数据完整性
            actual_size = len(value)
            stored_size = int(stored_size)
            
            if actual_size != stored_size:
                log.error(f"Data corruption detected for key {key}: expected size {stored_size}, got {actual_size}")
                return None

            return value
        except Exception as e:
            log.error(f"Failed to retrieve data from Redis: {e}")
            return None

    def delete(self, key: str) -> bool:
        """删除数据"""
        try:
            pipeline = self.redis_client.pipeline()
            pipeline.delete(key)
            pipeline.delete(f"{key}:size")
            pipeline.execute()
            return True
        except Exception as e:
            log.error(f"Failed to delete data from Redis: {e}")
            return False

    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        try:
            return bool(self.redis_client.exists(key))
        except Exception as e:
            log.error(f"Failed to check key existence in Redis: {e}")
            return False

    def keys(self, pattern: str = "*") -> List[str]:
        """获取匹配模式的所有键名
        
        Args:
            pattern: 匹配模式，默认为所有键
            
        Returns:
            List[str]: 匹配的键名列表
        """
        return [k.decode() for k in self.redis_client.keys(pattern)]

    def flush(self) -> bool:
        """清空当前数据库
        
        Returns:
            bool: 操作是否成功
        """
        try:
            self.redis_client.flushdb()
            return True
        except Exception:
            return False

    def close(self):
        """关闭 Redis 连接"""
        self.redis_client.close()

    def get_all_keys_with_prefix(self, prefix: str) -> list:
        """获取所有以指定前缀开头的键"""
        try:
            # 使用KEYS命令查找匹配的键（注意：在生产环境中应谨慎使用KEYS命令）
            keys = self.redis_client.keys(f"{prefix}*")
            return [key.decode('utf-8') if isinstance(key, bytes) else key for key in keys]
        except Exception as e:
            log.error(f"获取键前缀失败: {str(e)}")
            return []
