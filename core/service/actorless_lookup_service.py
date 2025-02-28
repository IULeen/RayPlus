from typing import Tuple, Optional, TYPE_CHECKING
import threading
import time
from collections import OrderedDict
from utils import logger
from utils.serialization import loads_Actorless
from config.setting import REDIS_HOST, REDIS_PORT, RAY_INIT_CONFIG
from core.state_manager.state_decision import StateDecisionManager

# 使用 TYPE_CHECKING 避免循环导入
if TYPE_CHECKING:
    from core.actor.actorless_decorator import Actorless

log = logger.get_logger(__name__)

class ActorlessLookupService:
    """Actorless 实例生命周期管理服务"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            # 本地缓存，用于存储活跃的 Actorless 实例
            self.local_cache = {}
            # 状态决策管理器
            self.state_manager = StateDecisionManager()
            self.initialized = True
            # 将free_cpus替换为free_memory，单位为MB
            self.free_memory = RAY_INIT_CONFIG.get("resouces", {"memory": 4096}).get("memory", 4096)  # 默认4GB内存
            # 记录总内存大小，用于日志和监控
            self.total_memory = self.free_memory
            # 使用 OrderedDict 来实现 LRU 缓存管理
            self.idle_actorless_container = OrderedDict()
            # 添加信号量来控制并发访问
            self._semaphore = threading.Semaphore(1)  # 使用单一信号量控制内存资源访问
            # 添加锁保护对 idle_actorless_container 的并发访问
            self._lock = threading.RLock()
            
    def start(self, actorless_id: str, actorless_ref: 'Actorless') -> bool:
        """
        尝试启动一个 actorless 实例
        
        Args:
            actorless_id: 实例的唯一ID
            actorless_ref: Actorless实例引用，用于获取内存大小
            
        Returns:
            bool: 是否成功获取到资源
        """
        # 获取实例需要的内存大小，默认为10MB
        memory_needed = getattr(actorless_ref, "_state_memory_size", 10)
        
        log.info(f"尝试为实例 {actorless_id} 启动容器, 当前可用内存: {self.free_memory}MB, 需要内存: {memory_needed}MB")
        
        # 检查当前实例是否已经在运行
        with self._lock:
            is_running = actorless_id in self.local_cache and self.local_cache[actorless_id].is_alive()
            if is_running:
                log.info(f"实例 {actorless_id} 已经在运行中，不需要额外资源")
                # 已经在运行的实例，不需要再获取资源
                return True
        
        # 尝试获取内存资源
        with self._semaphore:
            # 标记当前线程已持有信号量
            threading.current_thread()._actorless_semaphore_held = True
            
            if self.free_memory >= memory_needed:
                # 如果有足够内存，分配资源
                self.free_memory -= memory_needed
                
                # 如果成功获取资源，从空闲容器中移除（如果存在）
                with self._lock:
                    if actorless_id in self.idle_actorless_container:
                        # 已经在运行的实例被再次访问，从空闲列表中移除
                        self.idle_actorless_container.pop(actorless_id)
                
                log.info(f"成功为实例 {actorless_id} 分配内存资源: {memory_needed}MB, 剩余内存: {self.free_memory}MB")
                # 移除标记
                delattr(threading.current_thread(), '_actorless_semaphore_held')
                return True
            else:
                # 如果没有足够内存，尝试释放空闲的 actorless 实例
                log.info(f"没有足够内存资源给实例 {actorless_id}, 尝试释放空闲实例")
                
                with self._lock:
                    # 检查是否有空闲实例可以释放
                    if self.idle_actorless_container:
                        # 一次性释放多个实例直到有足够内存或没有更多实例可释放
                        released_count = 0
                        released_memory = 0
                        
                        # 创建待释放实例列表
                        instances_to_release = []
                        for lru_id, (lru_ref, _) in list(self.idle_actorless_container.items()):
                            instances_to_release.append((lru_id, lru_ref))
                            # 如果已经可以释放足够内存，就停止
                            lru_memory = getattr(lru_ref, "_state_memory_size", 10)
                            released_memory += lru_memory
                            if self.free_memory + released_memory >= memory_needed:
                                break
                        
                        log.info(f"尝试释放 {len(instances_to_release)} 个空闲实例，预计释放内存: {released_memory}MB")
                        
                        # 执行释放
                        for lru_id, lru_ref in instances_to_release:
                            # 增加ID一致性检查
                            if lru_ref._unique_id != lru_id:
                                log.error(f"LRU容器中ID不一致：键为 {lru_id}，但实例ID为 {lru_ref._unique_id}")
                                # 使用实例的实际ID
                                lru_id = lru_ref._unique_id
                            
                            # 从空闲容器中移除
                            if lru_id in self.idle_actorless_container:
                                self.idle_actorless_container.pop(lru_id)
                            
                            # 检查实例是否有效
                            if hasattr(lru_ref, "_actor_handle") and lru_ref._actor_handle is not None:
                                log.info(f"根据LRU算法释放空闲实例 {lru_id}")
                                # 修改：先从LRU容器中移除，再调用check_point，避免死锁
                                try:
                                    lru_ref.check_point()
                                except Exception as e:
                                    log.error(f"释放实例 {lru_id} 时check_point失败: {str(e)}")
                                    # 即使check_point失败，也要确保状态一致
                                    lru_ref._is_alive = False
                                    lru_ref._actor_handle = None
                            else:
                                log.warning(f"实例 {lru_id} 的actor_handle已经是None，直接释放资源")
                                # 确保状态一致
                                lru_ref._is_alive = False
                                # 从本地缓存中移除该实例，避免后续引用
                                if lru_id in self.local_cache:
                                    self.local_cache.pop(lru_id)
                            
                            # 释放实例资源
                            self.finish(lru_id, lru_ref)
                            released_count += 1
                        
                        log.info(f"成功释放了 {released_count} 个空闲实例，释放内存: {released_memory}MB")
                        
                        # 释放实例后再次检查是否有足够内存
                        if self.free_memory >= memory_needed:
                            # 分配内存资源
                            self.free_memory -= memory_needed
                            log.info(f"释放空闲实例后成功获取内存资源, 剩余内存: {self.free_memory}MB")
                            return True
                        else:
                            log.warning(f"释放 {released_count} 个实例后仍无法获取足够内存，需要: {memory_needed}MB, 可用: {self.free_memory}MB")
                    else:
                        log.warning(f"没有空闲实例可以释放")
                
                # 如果还是没有足够内存，则等待一段时间后重试
                log.warning(f"等待资源可用, 实例 {actorless_id} 处于等待状态")
                time.sleep(1)  # 等待1秒
                
                # 重试几次
                for i in range(5):
                    with self._semaphore:
                        if self.free_memory >= memory_needed:
                            self.free_memory -= memory_needed
                            log.info(f"等待后成功获取内存资源, 剩余内存: {self.free_memory}MB")
                            return True
                    time.sleep(2)  # 每次等待时间增加
                
                log.error(f"等待超时, 无法为实例 {actorless_id} 分配内存资源")
                return False

    def finish(self, actorless_id: str, actorless_ref: 'Actorless'):
        """释放一个实例的内存资源，确保状态一致性"""
        # 获取实例占用的内存大小
        memory_size = getattr(actorless_ref, "_state_memory_size", 10)
        
        # 查找实例并更新状态
        if actorless_id in self.local_cache:
            instance = self.local_cache[actorless_id]
            instance._is_alive = False  # 确保状态一致

        # 释放内存资源 - 修复可能的死锁问题
        # 检查当前线程是否已经持有信号量，如果已持有则不再尝试获取
        if not hasattr(threading.current_thread(), '_actorless_semaphore_held'):
            with self._semaphore:
                self.free_memory += memory_size
                # 确保不超过总内存
                if self.free_memory > self.total_memory:
                    log.warning(f"内存计数异常：释放后可用内存 {self.free_memory}MB 超过总内存 {self.total_memory}MB，重置为总内存")
                    self.free_memory = self.total_memory
                log.info(f"释放实例 {actorless_id} 的内存资源: {memory_size}MB, 当前可用内存: {self.free_memory}MB")
        else:
            # 直接更新内存，不尝试获取信号量
            self.free_memory += memory_size
            # 确保不超过总内存
            if self.free_memory > self.total_memory:
                log.warning(f"内存计数异常：释放后可用内存 {self.free_memory}MB 超过总内存 {self.total_memory}MB，重置为总内存")
                self.free_memory = self.total_memory
            log.info(f"释放实例 {actorless_id} 的内存资源: {memory_size}MB, 当前可用内存: {self.free_memory}MB")

    def update_access_time(self, actorless_id: str, actorless_ref: 'Actorless', keep_alive: bool = True):
        """
        更新 actorless 实例的访问时间，用于 LRU 算法
        
        Args:
            actorless_id: 实例的唯一ID
            actorless_ref: Actorless 实例引用
            keep_alive: 是否保持实例活跃
        """
        # 验证ID一致性
        if actorless_id != actorless_ref._unique_id:
            log.error(f"更新访问时间时ID不一致：请求ID为 {actorless_id}，但实例ID为 {actorless_ref._unique_id}")
            # 使用实例的实际ID，而非传入的ID
            actorless_id = actorless_ref._unique_id


        with self._lock:
            if keep_alive:
                # 如果需要保持活跃，添加/更新到空闲容器中
                current_time = time.time()
                self.idle_actorless_container[actorless_id] = (actorless_ref, current_time)
                log.debug(f"Updated access time for actorless {actorless_id}, total idle instances: {len(self.idle_actorless_container)}")
            elif actorless_id in self.idle_actorless_container:
                # 如果不保持活跃且在空闲容器中，则移除
                self.idle_actorless_container.pop(actorless_id)
                log.debug(f"Removed actorless {actorless_id} from idle container")

    def find_actorless_by_id(self, actorless_id: str) -> bool:
        """
        查找 Actorless 实例是否存活
        
        Args:
            actorless_id: 实例的唯一ID
            
        Returns:
            bool: 实例是否存活
        """
        if actorless_id in self.local_cache:
            instance = self.local_cache[actorless_id]
            is_alive = instance.is_alive() and instance._actor_handle is not None
            log.debug(f"在本地缓存中找到实例 {actorless_id}, is_alive = {is_alive}")
            return is_alive
        else:
            log.debug(f"实例 {actorless_id} 不在本地缓存中")
            return False

    def find_actorless(self, actor_function_name: str, actorless_id: str) -> Tuple[bool, Optional['Actorless']]:
        """查找 Actorless 实例，增加ID一致性检查"""
        # 1. 检查本地缓存
        if actorless_id in self.local_cache:
            instance = self.local_cache[actorless_id]
            # 确认实例ID与请求的ID一致
            if instance._unique_id != actorless_id:
                log.error(f"缓存不一致：请求ID为 {actorless_id}，但实例ID为 {instance._unique_id}")
                # 移除不一致的缓存项
                self.local_cache.pop(actorless_id)
            else:
                return True, instance
        
        # 2. 通过状态管理器查找
        success, actorless_ref = self.state_manager.load_state(actorless_id)
        if success:
            # 再次确认ID一致性
            if actorless_ref._unique_id != actorless_id:
                log.error(f"状态管理器返回错误实例：请求ID为 {actorless_id}，但实例ID为 {actorless_ref._unique_id}")
                return False, None
            # 找到实例后加入本地缓存
            self.local_cache[actorless_id] = actorless_ref
            return True, actorless_ref
        
        return False, None

    def store_actorless(self, unique_id: str, actorless_ref: 'Actorless'):
        """存储 Actorless 实例，确保ID一致性"""
        # 检查实例ID是否与传入的ID一致
        if actorless_ref._unique_id != unique_id:
            log.error(f"尝试以ID {unique_id} 存储实例，但实例ID为 {actorless_ref._unique_id}")
            # 使用实例自己的ID进行存储，而不是传入的ID
            unique_id = actorless_ref._unique_id
        
        # 更新本地缓存
        self.local_cache[unique_id] = actorless_ref
        # 通过状态管理器存储
        if not actorless_ref._is_alive:
            self.state_manager.store_state(unique_id, actorless_ref)
    
    def remove_actorless(self, unique_id: str):
        """移除 Actorless 实例"""
        # 1. 从本地缓存移除
        if unique_id in self.local_cache:
            self.local_cache.pop(unique_id)
        
        # 2. 从空闲容器中移除
        with self._lock:
            if unique_id in self.idle_actorless_container:
                self.idle_actorless_container.pop(unique_id)
        
        # 3. 通过状态管理器删除
        self.state_manager.delete_actorless(unique_id)
        
    def _search_actorless_instance(self, actor_function_name: str, actorless_id: str):
        # 使用查找服务
        found, instance = self.find_actorless(actor_function_name, actorless_id)
        if found and instance._unique_id != actorless_id:
            log.error(f"查找实例ID不匹配：请求 {actorless_id}，但返回 {instance._unique_id}")
            return False, None
        return found, instance
        