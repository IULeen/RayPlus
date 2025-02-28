# 本地库
from typing import Callable, Optional
import time

# ray 官方库
import ray

# 自定义第三方库
from utils.annotations import PublicAPI, Deprecated
from utils import logger
from core.actor.ActorlessConfig import ActorlessConfig

from core.service.actorless_lookup_service import ActorlessLookupService

log = logger.get_logger(__name__)


# 初始化全局单例服务
lookup_service = ActorlessLookupService()

"""
    old 注释: To be deleted
    现在实现了Actorless的装饰器，用于在两次调用之间停止 Actor实例，避免资源浪费
    TODO： 1. 完善一些细节，看有无其他需要添加的管理类
           2. 增加模块，使得能够支持 数据状态的持久化
    --------------------------------------------------------
    Actorless 包装类，用于延迟创建与自动管理 Actor 的生命周期，
    实现两次调用之间释放 Actor 资源、状态持久化与恢复。

    主要逻辑包括：
      - 利用 ray.remote 包装实际 Actor 类。
      - 通过调用 checkpoint 保存状态后停止 Actor。
      - 通过 recover 重建 Actor 并恢复状态。
"""
@PublicAPI(stability="alpha")
class Actorless:
    """
        Actorless 类变量详细说明：

        实例属性：
            _name (str):
                保存 Actorless 实例的名称。
            _namespace (str):
                存储命名空间信息，用于将不同实例进行逻辑分组或隔离管理。
            _is_alive (bool):
                标识当前 Actor 实例是否处于活跃状态，True 表示实例正在运行，False 则表示实例已停止或未启动。
            _unique_id:
                Actorless 实例的唯一标识符，初始为 None，后续可在bind()实例创建时赋予具体值，用于在系统中区分实例。
            _actor_config (ActorlessConfig):
                包含 Actorless 全部配置信息的对象，其中存储了反序列化的 Actor 定义以及其它相关配置参数。
            _actor_def:
                从 _actor_config 中获取的 Actor 定义，通过 ray.remote 包装后，用于创建远程 Actor 实例，确保其能在分布式环境中运行。
            _actor_handle:
                远程创建的 Actor 实例的句柄，用于后续方法调用和状态管理；当 Actor 被停止时，该值将被置为 None。
            _is_persistent (bool):
                标识当前 Actor 实例是否需要持久化存储，True 表示需要持久化存储，False 则表示不需要持久化存储。
            _is_always_alive (bool):
                标识当前 Actor 实例是否需要一直存活，True 表示需要一直存活，False 则表示不需要一直存活。
            _storage_level (tuple):
                用户自己限定 状态 多级存储层级范围，默认是 (0, 3)，分别表示 状态存储在 保活实例、本地内存、Redis、MinIO 中。
            _args:
                保存 Actor 实例初始化时的位置参数，便于在 Actor 重启或状态恢复时重新传递这些参数。
            _kwargs:
                保存 Actor 实例初始化时的关键字参数，同样用于在重启或状态恢复过程中重新设置实例状态。
            _state:
                用于存储 Actor 实例状态的变量，通常通过调用 Actor 内定义的 _store_states() 方法获得，便于后续进行状态持久化和恢复。
            _state_memory_size:
                Actor 状态占用的内存大小（MB）
    """
    def __init__(self, name: str, namespace: str, actorlessConfig: ActorlessConfig) -> None:
        # log.info("__init__ function is called!")
        # log.info(f"Name is {name}, namespace is {namespace}, actor_def is {actorlessConfig}")
        # log.info(f"Type of actor_def is {type(actorlessConfig.deployment_def)}")
        self._name = name
        self._namespace = namespace
        self._is_alive = False
        self._unique_id = None
        
        # 首先从 actorlessConfig 中获取 反序列化后的 actor_def
        # 然后再将 _actor_def 重新赋值为 被 ray.remote() 封装的 Actor 类模板
        self._actor_config = actorlessConfig
        self._actor_def = actorlessConfig.deployment_def
        self._actor_def = ray.remote(self._actor_def)
        
        self._actor_handle = None
        self._args = None
        self._kwargs = None

        # 用于保存 Actor 的状态
        # 目前需要用户手动实现 _store_states() 和 _recover_state() 方法
        self._state = None

        # 用于确认是否需要持久化存储
        self._is_persistent = False

        # 用于确认是否需要 一直存活, 默认是 False
        # 如果为 True，则 Actor 实例会一直存活，不会被停止。直到用户主动调用 delete() 方法
        self._is_always_alive = False

        # 用户自己限定 状态 多级存储层级范围
        # 默认是 (0, 3)，分别表示 状态存储在 保活实例、本地内存、Redis、MinIO 中
        self._storage_level = (0, 3)

        # 初始化状态内存大小
        self._state_memory_size = 10  # 默认状态内存大小：10MB


    def check_point(self):
        """保存 Actor 状态"""
        try:
            log.debug("Store actor's state")
            if self._actor_handle is  None:
                log.info("-------------------------Actor is not alive, no need to store state")
                return
            
            log.info(f"########################################################Store actor's state")
            try:
                # 尝试获取状态，但即使失败也要继续执行关闭操作
                self._state = ray.get(self._actor_handle._store_states.remote())
            except Exception as e:
                log.error(f"获取Actor状态失败: {str(e)}")
                # 状态获取失败，但仍然需要关闭Actor
            
            # 优雅地关闭 Actor
            if self._actor_handle:
                try:
                    ray.kill(self._actor_handle)
                except Exception as e:
                    log.error(f"关闭Actor失败: {str(e)}")
                finally:
                    # 无论如何都要更新状态
                    self._actor_handle = None
                    self._is_alive = False
                    log.info(f"Actor {self._unique_id} 已成功执行检查点操作并销毁")
            
            # 如果需要持久化，则存储状态
            if self._is_persistent:
                lookup_service.store_actorless(self._unique_id, self)
                log.debug(f"Actor {self._unique_id} 的状态已持久化")
                
        except Exception as e:
            log.error(f"Checkpoint failed: {str(e)}")
            # 确保状态一致
            self._actor_handle = None
            self._is_alive = False
            raise

    def recover(self):
        """恢复 Actor 状态"""
        try:
            # 使用 unique_id 作为命名空间的一部分，确保每个实例都是独立的
            # instance_namespace = f"{self._namespace}_{self._unique_id}"
            try:
                self._actor_handle = ray.get_actor(self._unique_id, namespace=self._namespace)
                self._is_alive = True
                log.debug(f"Found existing actor with ID {self._unique_id}")
            except Exception as e:
                log.debug(f"Creating new actor with ID {self._unique_id}")
                self._actor_handle = self._actor_def.options(
                    name=self._unique_id,
                    namespace=self._namespace,
                    lifetime="detached"
                ).remote(*self._args, **self._kwargs)
                
                if self._state is not None:
                    log.debug(f"Recovering state for actor {self._unique_id}")
                    self._actor_handle._recover_states.remote(self._state)
                    
                self._is_alive = True
                
        except Exception as e:
            log.error(f"Recovery failed for actor {self._unique_id}: {str(e)}")
            self._is_alive = False
            raise

    @Deprecated
    def _recover(self):
        if self._state is not None:
            self._actor_handle._recover_states.remote(self._state)


    def get_handle(self):
        return self._actor_handle
    

    def __call__(self, *args, **kwargs):
        # *args, **kwargs: 在这里没有作用
        # 本来它们是类的初始化参数，但Actorless装饰器禁止直接初始化原类
        raise RuntimeError(
            "Actorless cannot be called directly. "
            "Use `rayless.invoke() instead`"
        )

    def bind(self, unique_id: str=None,*args, **kwargs):
        # 创建当前对象的深拷贝
        import copy
        new_instance = copy.deepcopy(self)
        
        # 设置新实例的属性
        new_instance._unique_id = unique_id
        new_instance._args = args
        new_instance._kwargs = kwargs
        new_instance._is_alive = False
        new_instance._actor_handle = None
        
       
        return new_instance
    
    def options(self, unique_id: str=None, is_persistent: bool=False, is_always_alive: bool=False, storage_level: tuple=(0, 3), state_memory_size: int=None):
        # 只有当明确提供了unique_id值时才更新，避免用None覆盖已有的值
        if unique_id is not None:
            self._unique_id = unique_id
        self._is_persistent = is_persistent
        self._is_always_alive = is_always_alive
        self._storage_level = storage_level
        
        # 更新状态内存大小（如果提供）
        if state_memory_size is not None:
            self._state_memory_size = max(1, state_memory_size)  # 确保至少为1MB
        
        lookup_service.store_actorless(unique_id, self)
        return self
    
    def is_alive(self):
        return self._is_alive
    


@PublicAPI(stability="alpha")
def actorless(
    _func_or_class: Optional[Callable] = None,
    name = "SimpleActor",
    namespace = "default",
) -> Callable[[Callable], Actorless]:
    
    # # _func_or_class 是可以直接调用的函数和类
    # # 通过 ray.remote() 装饰器函数，可以将其转变为 Ray Task/Actor
    # myActor = ray.remote(_func_or_class)
    # actorInstance = myActor.options(name=name, namespace=namespace).remote(init_value=10)
    # result = ray.get(actorInstance.get_state.remote())
    
    def decorator(_func_or_class):

        # ActorlessConfig 主要用于将 _func_or_class 序列化 
        actorlessConfig = ActorlessConfig.create(_func_or_class)

        return Actorless(
            name if name is not None else _func_or_class.__name__,
            namespace if namespace is not None else namespace,
            actorlessConfig = actorlessConfig,
        )

    return decorator(_func_or_class) if callable(_func_or_class) else decorator


"""
 TODO: 暂时有一个逻辑错误，即传入的 actorless_ref 是传值，而不是传引用，对 actorless_ref 的修改不会影响原来的实例
"""


@PublicAPI(stability="alpha")
def invoke(actorless_ref: Actorless, keep_alive: bool, method_name: str, *args, is_sync:bool = True, **kwargs):
    # 首先检查实例是否有有效的ID
    if actorless_ref._unique_id is None:
        log.error("调用失败: Actorless实例没有有效的unique_id，请确保先调用bind方法设置ID")
        raise ValueError("Actorless实例没有有效的unique_id，请确保先调用bind方法设置ID")
    
    # 有一个全局数据库，可以查询 actorless_ref 对应的 actor 实例是否存活
    is_Alive = lookup_service.find_actorless_by_id(actorless_ref._unique_id)
    log.info(f"[invoke] 实例ID: {actorless_ref._unique_id}, 名称: {actorless_ref._name}, is_Alive: {is_Alive}")
    
    # 调用前始终尝试获取资源，不管实例是否处于活跃状态
    # 只有已经在运行且保持活跃的实例才不需要重新获取资源
    need_resource = not is_Alive or actorless_ref._actor_handle is None
    log.info(f"[invoke] 实例ID: {actorless_ref._unique_id}, need_resource: {need_resource}, handle: {actorless_ref._actor_handle is not None}")
    
    if need_resource:
        # 尝试获取资源，如果没有可用资源，会根据LRU算法释放空闲实例或等待
        resource_acquired = lookup_service.start(actorless_ref._unique_id, actorless_ref)
        log.info(f"[invoke] 实例ID: {actorless_ref._unique_id}, 获取资源结果: {resource_acquired}")
        
        if not resource_acquired:
            log.error(f"[invoke] 获取资源失败! 实例ID: {actorless_ref._unique_id}")
            raise Exception(f"Failed to acquire resource for actorless {actorless_ref._unique_id}")
    
    # 只有在需要恢复的情况下才调用recover
    if not actorless_ref.is_alive() or actorless_ref._actor_handle is None:
        actorless_ref.recover()
        log.info(f"[invoke] 实例ID: {actorless_ref._unique_id}, 恢复完成, is_alive={actorless_ref.is_alive()}")
    
    actor_handle = actorless_ref.get_handle()
    log.debug(f"实例ID: {actorless_ref._unique_id}, Actor_handle: {actor_handle}")

    # 执行开始时间
    start_time = time.time()
    
    method = getattr(actor_handle, method_name, None)

    if method is not None:
        res = method.remote(*args, **kwargs)
    else:
        # 如果方法不存在，需要释放资源
        if need_resource and not keep_alive:
            lookup_service.finish(actorless_ref._unique_id)
        raise Exception(f"Method {method_name} not found")

    # 定义该次调用是否是异步调用，默认为否，需要等待函数执行返回结果
    if is_sync:
        try:
            res = ray.get(res)
        except Exception as e:
            # 发生异常时，如果不保活，需要释放资源
            if need_resource and not keep_alive:
                lookup_service.finish(actorless_ref._unique_id)
            raise e
    else:
        res = {"Invoke Success!"}
    
    # 更新actorless的访问时间，用于LRU算法
    lookup_service.update_access_time(actorless_ref._unique_id, actorless_ref, keep_alive)

    if not keep_alive:
        # 执行检查点并停止实例
        actorless_ref.check_point()
        # 不保活才需要释放资源，且只有之前获取了资源才需要释放
        if need_resource:
            lookup_service.finish(actorless_ref._unique_id, actorless_ref)
        lookup_service.store_actorless(actorless_ref._unique_id, actorless_ref)
        actorless_ref = None
    else:
        lookup_service.store_actorless(actorless_ref._unique_id, actorless_ref)

    # 记录执行耗时
    log.debug(f"Method {method_name} execution time: {time.time() - start_time:.4f}s")
    
    return res


# TODO: Modify, actorless_ref 严格来说是一个 Actorless 实例，在 invoke 时，应该只需要 actor_id 就可以调用恢复 actor_ref
#           然后再调用其中的method方法
@PublicAPI(stability="alpha")
def invoke_by_actorlessID(actor_function_name: str, actorless_id: str, keep_alive: bool, method_name: str, *args, is_sync:bool = True, **kwargs):

    # TODO
    queryRes, actorless_ref = _search_actorless_instance(actor_function_name, actorless_id)
 
    if queryRes == False:
        log.error(f"[invoke_by_actorlessID] 未找到Actorless实例: {actorless_id}")
        raise Exception(f"Actorless {actorless_id} not found")
    else:
        # 有一个全局数据库，可以查询 actorless_ref 对应的 actor 实例是否存活
        # 目前有点问题，按道理来说不应该使用 实例化后的 actorlessRef 查询
        if actorless_ref._unique_id != actorless_id:
            log.error(f"[invoke_by_actorlessID] 实例ID不匹配，期望：{actorless_id}，实际：{actorless_ref._unique_id}")
            raise Exception(f"实例ID不匹配")

        is_Alive = actorless_ref.is_alive()
        log.info(f"[invoke_by_actorlessID] 实例 {actorless_id} is_Alive: {is_Alive}")

        # 调用前始终尝试获取资源，不管实例是否处于活跃状态
        # 只有已经在运行且保持活跃的实例才不需要重新获取资源
        need_resource = not is_Alive or actorless_ref._actor_handle is None
        log.info(f"[invoke_by_actorlessID] 实例 {actorless_id} need_resource: {need_resource}")
        
        if need_resource:
            # 尝试获取资源，如果没有可用资源，会根据LRU算法释放空闲实例或等待
            resource_acquired = lookup_service.start(actorless_id, actorless_ref)
            log.info(f"[invoke_by_actorlessID] 实例 {actorless_id} 获取资源结果: {resource_acquired}")
            
            if not resource_acquired:
                log.error(f"[invoke_by_actorlessID] 获取资源失败! 实例: {actorless_id}")
                raise Exception(f"Failed to acquire resource for actorless {actorless_id}")

            # 只有在需要恢复的情况下才调用recover
            if not actorless_ref.is_alive() or actorless_ref._actor_handle is None:
                actorless_ref.recover()
                log.info(f"[invoke_by_actorlessID] 实例 {actorless_id} 恢复完成")

        actor_handle = actorless_ref.get_handle()
        log.debug(f"Actor_handle is: {actor_handle}")

        method = getattr(actor_handle, method_name, None)

        start_time = time.time()

        if method is not None:
            res = method.remote(*args, **kwargs)
        else:
            # 如果方法不存在，需要释放资源
            if need_resource and not keep_alive:
                lookup_service.finish(actorless_id)
            raise Exception(f"Method {method_name} not found")

        if is_sync:
            try:
                res = ray.get(res)
            except Exception as e:
                # 发生异常时，如果不保活，需要释放资源
                if need_resource and not keep_alive:
                    lookup_service.finish(actorless_id)
                raise e
        else:
            res = {"Invoke Success!"}

        # 更新actorless的访问时间，用于LRU算法
        lookup_service.update_access_time(actorless_id, actorless_ref, keep_alive)

        if not keep_alive:
            actorless_ref.check_point()
            # check_point() 函数中，已经将 _actor_handle 置为 None，也将_is_alive置为False
            # 不保活才需要释放资源，且只有之前获取了资源才需要释放
            if need_resource:
                lookup_service.finish(actorless_id)
            lookup_service.store_actorless(actorless_ref._unique_id, actorless_ref)
            actorless_ref = None
        else:
            lookup_service.store_actorless(actorless_ref._unique_id, actorless_ref)

        # 记录执行耗时
        log.debug(f"Method {method_name} execution time: {time.time() - start_time:.4f}s")

        return res


@PublicAPI(stability="alpha")
def invoke_by_http(actor_function_name: str, actorless_id: str, keep_alive: bool, method_name: str, *args, is_sync:bool = True, **kwargs):   
    # TODO
    queryRes, actorless_ref = _search_actorless_instance(actor_function_name, actorless_id)

    if queryRes == False:
        log.info(f"[invoke_by_http] 未找到Actorless实例: {actorless_id}，尝试通过Ray获取")
        try:
            actor_handle = ray.get_actor(actorless_id, namespace="default_"+actorless_id)

            if actor_handle is not None:
                log.debug(f"Actor_handle is: {actor_handle}")

                method = getattr(actor_handle, method_name, None)

                if method is not None:
                    res = method.remote(*args, **kwargs)
                else:
                    raise Exception(f"Method {method_name} not found")
                    
                if is_sync:
                    res = ray.get(res)
                else:
                    res = {"Invoke Success!"}
                    
                return res
            
        except Exception as e:
            log.error(f"[invoke_by_http] 通过Ray获取Actorless实例失败: {actorless_id}, 错误: {str(e)}")
            raise Exception(f"Actorless {actorless_id} not found: {str(e)}")
    else:
        # 有一个全局数据库，可以查询 actorless_ref 对应的 actor 实例是否存活
        is_Alive = actorless_ref.is_alive()
        log.info(f"[invoke_by_http] 实例 {actorless_id} is_Alive: {is_Alive}")

        # 调用前始终尝试获取资源，不管实例是否处于活跃状态
        # 只有已经在运行且保持活跃的实例才不需要重新获取资源
        need_resource = not is_Alive or actorless_ref._actor_handle is None
        log.info(f"[invoke_by_http] 实例 {actorless_id} need_resource: {need_resource}")
        
        if need_resource:
            # 尝试获取资源，如果没有可用资源，会根据LRU算法释放空闲实例或等待
            resource_acquired = lookup_service.start(actorless_id, actorless_ref)
            log.info(f"[invoke_by_http] 实例 {actorless_id} 获取资源结果: {resource_acquired}")
            
            if not resource_acquired:
                log.error(f"[invoke_by_http] 获取资源失败! 实例: {actorless_id}")
                raise Exception(f"Failed to acquire resource for actorless {actorless_id}")

            # 只有在需要恢复的情况下才调用recover
            if not actorless_ref.is_alive() or actorless_ref._actor_handle is None:
                actorless_ref.recover()
                log.info(f"[invoke_by_http] 实例 {actorless_id} 恢复完成")

        actor_handle = actorless_ref.get_handle()
        log.debug(f"Actor_handle is: {actor_handle}")

        method = getattr(actor_handle, method_name, None)

        start_time = time.time()

        if method is not None:
            res = method.remote(*args, **kwargs)
        else:
            # 如果方法不存在，需要释放资源
            if need_resource and not keep_alive:
                lookup_service.finish(actorless_id)
            raise Exception(f"Method {method_name} not found")

        if is_sync:
            try:
                res = ray.get(res)
            except Exception as e:
                # 发生异常时，如果不保活，需要释放资源
                if need_resource and not keep_alive:
                    lookup_service.finish(actorless_id)
                raise e
        else:
            res = {"Invoke Success!"}

        # 更新actorless的访问时间，用于LRU算法
        lookup_service.update_access_time(actorless_id, actorless_ref, keep_alive)

        if not keep_alive:
            actorless_ref.check_point()
            # check_point() 函数中，已经将 _actor_handle 置为 None，也将_is_alive置为False
            # 不保活才需要释放资源，且只有之前获取了资源才需要释放
            if need_resource:
                lookup_service.finish(actorless_id)
            lookup_service.store_actorless(actorless_ref._unique_id, actorless_ref)
            actorless_ref = None
        else:
            lookup_service.store_actorless(actorless_ref._unique_id, actorless_ref)

        # 记录执行耗时
        log.debug(f"Method {method_name} execution time: {time.time() - start_time:.4f}s")

        return res

# 从全局查找模块中获取 Actor 实例
# TODO：该函数是否应该在这里呢？
def _search_actorless_instance(actor_function_name: str, actorless_id: str):
    # 使用查找服务
    return lookup_service.find_actorless(actor_function_name, actorless_id)



@PublicAPI(stability="alpha")
def deleteActorless(actorless_ref: Actorless):
    _delete_actorless_instance(actorless_ref)

def _delete_actorless_instance(actorless_ref: Actorless):
    unique_id = actorless_ref._unique_id
    lookup_service.remove_actorless(unique_id)