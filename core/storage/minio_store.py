from typing import Optional, List
from minio import Minio
from minio.error import S3Error
import io
import logging
from config.setting import (
    MINIO_HOST,
    MINIO_PORT,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
    MINIO_BUCKET_NAME
)

log = logging.getLogger(__name__)

class MinIOStorage:
    """MinIO 存储管理类
    
    提供基于 MinIO 的对象存储功能，作为多级存储的持久化层
    """
    
    def __init__(
        self,
        host: str = MINIO_HOST,
        port: int = MINIO_PORT,
        access_key: str = MINIO_ACCESS_KEY,
        secret_key: str = MINIO_SECRET_KEY,
        bucket_name: str = MINIO_BUCKET_NAME,
        secure: bool = False
    ):
        """初始化 MinIO 客户端
        
        Args:
            host: MinIO 服务器地址
            port: MinIO 服务器端口
            access_key: 访问密钥
            secret_key: 秘密密钥
            bucket_name: 存储桶名称
            secure: 是否使用 HTTPS
        """
        self.bucket_name = bucket_name
        endpoint = f"{host}:{port}"
        
        try:
            self.client = Minio(
                endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=secure
            )
            
            # 确保存储桶存在
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                log.info(f"Created bucket: {bucket_name}")
                
        except Exception as e:
            log.error(f"Failed to initialize MinIO client: {e}")
            raise
            
    def put(self, key: str, data: bytes, metadata: Optional[dict] = None) -> bool:
        """存储数据到 MinIO
        
        Args:
            key: 对象键名
            data: 要存储的数据
            metadata: 对象元数据
            
        Returns:
            bool: 是否存储成功
        """
        try:
            # 将字节数据转换为文件对象
            data_stream = io.BytesIO(data)
            data_size = len(data)
            
            # 上传数据
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=key,
                data=data_stream,
                length=data_size,
                metadata=metadata
            )
            
            log.debug(f"Successfully stored object {key} to MinIO")
            return True
            
        except Exception as e:
            log.error(f"Failed to store object {key} to MinIO: {e}")
            return False
            
    def get(self, key: str) -> Optional[bytes]:
        """从 MinIO 获取数据
        
        Args:
            key: 对象键名
            
        Returns:
            Optional[bytes]: 获取的数据，如果不存在则返回 None
        """
        try:
            # 获取对象
            response = self.client.get_object(
                bucket_name=self.bucket_name,
                object_name=key
            )
            
            # 读取数据
            data = response.read()
            response.close()
            
            log.debug(f"Successfully retrieved object {key} from MinIO")
            return data
            
        except S3Error as e:
            if e.code == 'NoSuchKey':
                log.debug(f"Object {key} not found in MinIO")
                return None
            log.error(f"Failed to retrieve object {key} from MinIO: {e}")
            return None
            
        except Exception as e:
            log.error(f"Failed to retrieve object {key} from MinIO: {e}")
            return None
            
    def delete(self, key: str) -> bool:
        """从 MinIO 删除数据
        
        Args:
            key: 对象键名
            
        Returns:
            bool: 是否删除成功
        """
        try:
            self.client.remove_object(
                bucket_name=self.bucket_name,
                object_name=key
            )
            log.debug(f"Successfully deleted object {key} from MinIO")
            return True
            
        except Exception as e:
            log.error(f"Failed to delete object {key} from MinIO: {e}")
            return False
            
    def exists(self, key: str) -> bool:
        """检查对象是否存在
        
        Args:
            key: 对象键名
            
        Returns:
            bool: 对象是否存在
        """
        try:
            self.client.stat_object(
                bucket_name=self.bucket_name,
                object_name=key
            )
            return True
            
        except S3Error as e:
            if e.code == 'NoSuchKey':
                return False
            log.error(f"Failed to check object {key} existence in MinIO: {e}")
            return False
            
    def list_objects(self, prefix: str = "") -> List[str]:
        """列出指定前缀的所有对象
        
        Args:
            prefix: 对象键名前缀
            
        Returns:
            List[str]: 对象键名列表
        """
        try:
            objects = self.client.list_objects(
                bucket_name=self.bucket_name,
                prefix=prefix
            )
            return [obj.object_name for obj in objects]
            
        except Exception as e:
            log.error(f"Failed to list objects with prefix {prefix}: {e}")
            return []
            
    def get_metadata(self, key: str) -> Optional[dict]:
        """获取对象元数据
        
        Args:
            key: 对象键名
            
        Returns:
            Optional[dict]: 对象元数据
        """
        try:
            stat = self.client.stat_object(
                bucket_name=self.bucket_name,
                object_name=key
            )
            return stat.metadata
            
        except Exception as e:
            log.error(f"Failed to get metadata for object {key}: {e}")
            return None
            
    def get_size(self, key: str) -> Optional[int]:
        """获取对象大小
        
        Args:
            key: 对象键名
            
        Returns:
            Optional[int]: 对象大小（字节）
        """
        try:
            stat = self.client.stat_object(
                bucket_name=self.bucket_name,
                object_name=key
            )
            return stat.size
            
        except Exception as e:
            log.error(f"Failed to get size for object {key}: {e}")
            return None
