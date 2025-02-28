from typing import Optional, Any, Tuple, List, TYPE_CHECKING
from enum import Enum
import redis
from dataclasses import dataclass
import sys
import os

from ray import cloudpickle as pickle

from utils import logger
from core.storage.redis_store import RedisStorage
from config.setting import REDIS_HOST, REDIS_PORT

# 使用 TYPE_CHECKING 避免循环导入
if TYPE_CHECKING:
    from core.actor.actorless_decorator import Actorless

log = logger.get_logger(__name__)

class StorageLevel(Enum):
    """存储层级枚举"""
    PROCESS = 0    # 存活进程（L0）
    MEMORY = 1     # 本地内存（L1）
    REDIS = 2      # Redis缓存（L2）
    MINIO = 3      # MinIO存储（L3）

@dataclass
class StorageMetrics:
    """存储层级性能指标"""
    read_latency: float  # 读取延迟(ms)
    write_latency: float # 写入延迟(ms)
    cost_per_gb: float  # 每GB存储成本($/GB/month)
    availability: float # 可用性(0-1)

# 定义各存储层级的性能指标
STORAGE_METRICS = {
    StorageLevel.PROCESS: StorageMetrics(0.001, 0.001, 0, 0.99),
    StorageLevel.MEMORY: StorageMetrics(0.1, 0.1, 0.1, 0.99),
    StorageLevel.REDIS: StorageMetrics(1.0, 1.0, 0.5, 0.999),
    StorageLevel.MINIO: StorageMetrics(10.0, 20.0, 0.02, 0.9999)
}

class StorageDecision:
    """存储决策结果"""
    def __init__(self, level: StorageLevel, ttl: Optional[int] = None):
        self.level = level
        self.ttl = ttl

class StateDecisionManager:
    """状态决策管理器，负责多级存储系统的决策和管理"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.redis_storage = RedisStorage(
                host=REDIS_HOST,
                port=REDIS_PORT,
                connection_pool=redis.ConnectionPool(
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                    max_connections=10
                )
            )
            # TODO: 初始化 MinIO 客户端
            self.initialized = True
    

    """
        TODO: Actor State 多级存储决策算法，后续有优化时可以修改此算法

        状态决策管理器，负责多级存储系统的决策和管理
        1. 根据 Actorless 实例的配置，决定状态存储级别
        2. 根据状态大小，决定存储层级
        3. 根据访问频率，决定存储层级
        4. 根据存储层级，决定存储位置
        5. 根据存储位置，决定存储方式
        6. 根据存储方式，决定存储策略
    """
    def decide_storage_level(self, actorless_ref: 'Actorless', state_size: Optional[int] = None) -> StorageDecision:
        """
        决定状态存储级别
        
        Args:
            actorless_id: Actorless 实例 ID
            state_size: 状态大小（字节）
            
        Returns:
            StorageDecision: 存储决策结果
        """
        try:
            # 获取 Actorless 实例配置
            actorless = actorless_ref
            if not actorless:
                log.warning(f"Actorless {actorless_ref} not found in cache, using default storage level")
                return StorageDecision(StorageLevel.REDIS, ttl=3600)
            
            # 1. 检查是否需要持续保活
            if actorless._is_always_alive:
                return StorageDecision(StorageLevel.PROCESS)
            
            # 2. 检查是否指定了固定存储层级
            storage_range = actorless._storage_level
            if storage_range[0] == storage_range[1]:
                return StorageDecision(StorageLevel(storage_range[0]))
            
            # 3. 多因素决策
            available_levels = [
                level for level in StorageLevel 
                if storage_range[0] <= level.value <= storage_range[1]
            ]
            
            # 如果没有状态大小信息，估算大小
            if state_size is None:
                state_size = self._estimate_state_size(actorless)
            
            # 计算每个可用存储层级的得分
            scores = self._calculate_storage_scores(
                available_levels,
                state_size=state_size,
                access_frequency=self._get_access_frequency(actorless_ref._unique_id)
            )
            
            # 选择得分最高的存储层级
            best_level = max(scores.items(), key=lambda x: x[1])[0]
            log.info(f"Best storage level: {best_level}")
            
            # 设置 TTL
            # 如果 _is_persistent 为 True，则设置 TTL 为 0，表示永久存储
            # 否则，根据状态大小和存储层级计算 TTL
            if actorless._is_persistent:
                ttl = 0
            else:
                ttl = self._calculate_ttl(best_level, state_size)
            
            return StorageDecision(best_level, ttl)
        
        except Exception as e:
            log.error(f"Error in storage decision for {actorless_ref._unique_id}: {e}")
            # 发生错误时使用默认配置
            return StorageDecision(StorageLevel.REDIS, ttl=3600)


    def _calculate_storage_scores(
        self, 
        available_levels: List[StorageLevel],
        state_size: int,
        access_frequency: float
    ) -> dict:
        """计算每个存储层级的得分"""
        scores = {}
        size_gb = state_size / (1024 * 1024 * 1024)  # 转换为GB
        
        for level in available_levels:
            metrics = STORAGE_METRICS[level]
            
            # 计算各项指标的权重得分
            latency_score = 1.0 / (metrics.read_latency * access_frequency)
            cost_score = 1.0 / (metrics.cost_per_gb * size_gb + 0.001)  # 避免除零
            availability_score = metrics.availability * 100
            
            # 根据状态大小调整权重
            if state_size < 1024 * 1024:  # 1MB以下
                weights = (0.5, 0.2, 0.3)  # 偏向低延迟
            elif state_size < 100 * 1024 * 1024:  # 100MB以下
                weights = (0.3, 0.4, 0.3)  # 平衡配置
            else:
                weights = (0.2, 0.5, 0.3)  # 偏向低成本
            
            # 计算综合得分
            scores[level] = (
                latency_score * weights[0] +
                cost_score * weights[1] +
                availability_score * weights[2]
            )
        
        return scores
    
    def _estimate_state_size(self, actorless) -> int:
        """估算状态大小"""
        try:
            # 使用 sys.getsizeof 估算状态大小
            return sys.getsizeof(actorless._state)
        except Exception:
            # 默认估算值：1MB
            return 1024 * 1024
    
    def _get_access_frequency(self, actorless_id: str) -> float:
        """获取状态访问频率（次/秒）"""
        # TODO: 实现访问频率统计
        # 当前返回默认值
        return 1.0
    
    def _calculate_ttl(self, level: StorageLevel, state_size: int) -> Optional[int]:
        """计算 TTL（秒）"""
        if level == StorageLevel.PROCESS:
            return None
        elif level == StorageLevel.MEMORY:
            return 300  # 5分钟
        elif level == StorageLevel.REDIS:
            if state_size < 1024 * 1024:  # 1MB以下
                return 3600  # 1小时
            else:
                return 1800  # 30分钟
        else:  # MINIO
            return None  # 永久存储
    
    def store_state(self, actorless_id: str, state: 'Actorless') -> bool:
        """
        存储状态到合适的存储层
        
        Args:
            actorless_id: Actorless 实例 ID
            state: 要存储的状态
            
        Returns:
            bool: 是否存储成功
        """
        # 添加ID一致性检查
        if actorless_id != state._unique_id:
            log.error(f"ID不一致：尝试以ID {actorless_id} 存储实例，但实例ID为 {state._unique_id}")
            # 使用实例的实际ID进行存储
            actorless_id = state._unique_id
        
        # Ray+ 中存储决策直接给定
        # decision = self.decide_storage_level(state)
        decision = StorageDecision(StorageLevel.REDIS, ttl=3600)

        try:
            # 确保即使StorageLevel为PROCESS也至少在Redis中保存一份
            # 这样可以防止状态完全丢失
            # TODO：这里重复存了两次，可以优化
            # serialized_state = _dumps_Actorless(state)
            # success = self.redis_storage.set(actorless_id, serialized_state, expire=3600)  # 默认至少存储1小时

            if decision.level == StorageLevel.REDIS:
                # 序列化状态数据
                serialized_state = _dumps_Actorless(state)
                return self.redis_storage.set(actorless_id, serialized_state, expire=decision.ttl)
            elif decision.level == StorageLevel.MINIO:
                # TODO: 实现 MinIO 存储
                pass
            return False
        except Exception as e:
            log.error(f"Failed to store state for {actorless_id}: {e}")
            return False
    
    def load_state(self, actorless_id: str) -> Tuple[bool, Optional[Any]]:
        """
        从存储层加载状态，增强错误诊断
        
        Args:
            actorless_id: Actorless 实例 ID
            
        Returns:
            Tuple[bool, Optional[Any]]: (是否成功, 状态数据)
        """
        try:
            # 详细日志记录和错误处理
            log.debug(f"尝试加载实例 {actorless_id} 的状态")

            # 反序列化前确保路径正确
            sys.path.append(os.path.join(os.path.dirname(__file__), "../../tests"))

            # 尝试从Redis读取
            if self.redis_storage.exists(actorless_id):
                log.info(f"Load state from Redis for {actorless_id}")
                try:
                    serialized_state = self.redis_storage.get(actorless_id)
                    if not serialized_state:
                        log.warning(f"Redis中存在键 {actorless_id} 但数据为空")
                        return False, None
                          
                    state = _loads_Actorless(serialized_state)
                    
                    # 反序列化后验证ID一致性
                    if state._unique_id != actorless_id:
                        log.error(f"从Redis加载的实例ID与请求的ID不匹配：加载ID为 {state._unique_id}，请求ID为 {actorless_id}")
                        return False, None
                    return True, state
                except Exception as e:
                    log.error(f"处理Redis状态时出错 {actorless_id}: {str(e)}")
                    return False, None
            
            # 2. 尝试从 MinIO 读取
            # TODO: 实现 MinIO 读取
            
            log.info(f"State not found for {actorless_id}")
            return False, None
            
        except Exception as e:
            log.error(f"Failed to load state for {actorless_id}: {e}")
            return False, None
    
    def delete_actorless(self, actorless_id: str) -> bool:
        """
        从所有存储层中删除 Actorless 实例的状态
        
        Args:
            actorless_id: Actorless 实例 ID
            
        Returns:
            bool: 是否删除成功
        """
        success = True
        
        # 1. 从 Redis 中删除
        try:
            if self.redis_storage.exists(actorless_id):
                success &= self.redis_storage.delete(actorless_id)
        except Exception as e:
            log.error(f"Failed to delete from Redis for {actorless_id}: {e}")
            success = False
            
        # 2. 从 MinIO 中删除
        # TODO: 实现 MinIO 删除
        
        return success
    
"""
    Actorless Class 的 Pickle 序列化与反序列化逻辑暂时先放在本文件中
    # 暂时先用 ray cloudpickle解决了，后续可以自定参考一下它的实现方式

    TODO: 后续该模块需要迁移到 Actorless 的 state_manager 模块中
    TODO: 多级存储，先从 ray object store 中查找与存储，如果找不到，再从 redis 中查找与存储
    TODO: 多级存储，存储位置的决策根据 Actorless 配置以及函数异构性来决定
"""
# 尝试性 pickle Actorless
def _dumps_Actorless(actorless_ref):
    """序列化 Actorless 对象，增强状态捕获"""
    try:
        # 保存所有关键状态
        data = {
            'state': getattr(actorless_ref, '_state', None),
            'name': getattr(actorless_ref, '_name', None),
            'namespace': getattr(actorless_ref, '_namespace', None),
            'unique_id': getattr(actorless_ref, '_unique_id', None),
            'args': getattr(actorless_ref, '_args', None),
            'kwargs': getattr(actorless_ref, '_kwargs', None),
            'is_alive': getattr(actorless_ref, '_is_alive', False),
            'is_persistent': getattr(actorless_ref, '_is_persistent', False),
            'is_always_alive': getattr(actorless_ref, '_is_always_alive', False),
            'storage_level': getattr(actorless_ref, '_storage_level', (0, 3)),
            # 添加更多可能的状态...
        }
        
        # 保存类相关信息
        if hasattr(actorless_ref, '_actor_config') and actorless_ref._actor_config:
            actor_config = actorless_ref._actor_config
            if hasattr(actor_config, 'deployment_def'):
                data['actor_class'] = actor_config.deployment_def
                data['actor_class_name'] = actor_config.deployment_def.__name__.encode('utf-8')
        
        # 记录序列化信息
        log.debug(f"序列化实例 {data['unique_id']}, 状态大小: {sys.getsizeof(data['state'])}")
        
        # 序列化
        return pickle.dumps(data)
    except Exception as e:
        log.error(f"序列化实例时出错: {e}")
        raise e

# 从redis中读取 serialized_actorless，并反序列化
def _loads_Actorless(serialized_actorless: bytes):
    """反序列化 Actorless 对象"""
    try:
        # 反序列化基本状态
        data = pickle.loads(serialized_actorless)
        
        # 重新导入必要的类
        from core.actor.actorless_decorator import Actorless, ray
        from core.actor.ActorlessConfig import ActorlessConfig
        
        # 获取原始类
        actor_class = data['actor_class']
        
        # 创建新的 ActorlessConfig
        config = ActorlessConfig.create(actor_class)
        
        # 创建新的 Actorless 实例
        actorless = Actorless(
            name=data['name'],
            namespace=data['namespace'],
            actorlessConfig=config
        )
        
        # 恢复状态
        actorless._state = data['state']
        actorless._unique_id = data['unique_id']
        actorless._args = data['args']
        actorless._kwargs = data['kwargs']
        actorless._is_alive = data['is_alive']

        # 恢复新增的状态
        actorless._is_persistent = data['is_persistent']
        actorless._is_always_alive = data['is_always_alive']
        actorless._storage_level = data['storage_level']
        
        # 确保 actor_def 被正确创建
        actorless._actor_def = ray.remote(actor_class)
        
        return actorless
    except Exception as e:
        log.error(f"Error occurred when deserializing actorless: {e}")
        raise e
