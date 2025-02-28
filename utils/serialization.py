import io
import logging
from typing import Any
import redis

import ray.cloudpickle as pickle
from config.setting import REDIS_HOST, REDIS_PORT

logger = logging.getLogger(__name__)


def pickle_dumps(obj: Any, error_msg: str):
    """Wrap cloudpickle.dumps to provide better error message
    when the object is not serializable.
    """
    try:
        return pickle.dumps(obj)
    except TypeError as e:
        sio = io.StringIO()
        # inspect_serializability(obj, print_file=sio)
        logger.error("pickle_dumps error")
        msg = f"{error_msg}:\n{sio.getvalue()}"
        raise TypeError(msg) from e
    
"""
    Actorless Class 的 Pickle 序列化与反序列化逻辑暂时先放在本文件中
    # 暂时先用 ray cloudpickle解决了，后续可以自定参考一下它的实现方式

    TODO: 后续该模块需要迁移到 Actorless 的 state_manager 模块中
    TODO: 多级存储，先从 ray object store 中查找与存储，如果找不到，再从 redis 中查找与存储
    TODO: 多级存储，存储位置的决策根据 Actorless 配置以及函数异构性来决定
"""
# 尝试性 pickle Actorless
def dumps_Actorless(actorless_ref, actorless_id):
    serialized_actorless = pickle.dumps(actorless_ref)
    # # 保存为本地文件
    # with open("actorless.pkl", "wb") as f:
    #     f.write(serialized_actorless)
    
    # # return 文件名称
    # return "actorless.pkl"

    # 保存到 redis 中
    
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    redis_client.set(actorless_id, serialized_actorless)

# 从redis中读取 serialized_actorless，并反序列化
def loads_Actorless(actorless_id: str):
    try:
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

        if redis_client.exists(actorless_id):
            serialized_actorless = redis_client.get(actorless_id)
        else:
            raise Exception("Actorless not found in redis")

        # with open(actorless_id, "rb") as f:
        #     serialized_actorless = f.read()

        deserialized_actorless = pickle.loads(serialized_actorless)
    except Exception as e:
        logger.error(f"Error occurred when loading actorless from redis: {e}")
        raise e

    return deserialized_actorless
