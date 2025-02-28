import sys
import os
import time
import ray
import logging
# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
# 添加这一行确保能访问到同级目录的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# 初始化 Ray，使用 working_dir 而不是 py_modules
# ray.init(runtime_env={
#     "working_dir": project_root,
#     "py_modules": [project_root]
# })

from utils import logger
from core.actor.actorless_decorator import invoke, invoke_by_actorlessID

log = logger.get_logger(__name__)

from GamePlayerActor import GamePlayerActor

def testPlayerActor(player_number: str):
    name = f"Player{player_number}"
    keep_alive = True

    player_ref = GamePlayerActor.bind(unique_id = name, player_id=name)
    player_ref.options(unique_id=name, is_persistent=False, storage_level=(0, 0), state_memory_size=10)
    

    # 测试 get_info 方法
    begin_time = time.time()
    for i in range(1):
        invocation_start_time = time.time()
        try:
            res = invoke(player_ref, keep_alive, "get_info")
            invocation_end_time = time.time()
            log.info(f"Time Consumption: {invocation_end_time - invocation_start_time}, Info: {res}")
            # 再次验证返回的player_id
            if res[0] != name:
                log.error(f"Expected player_id {name}, but got {res[0]}")
        except Exception as e:
            log.error(f"调用Player{player_number}失败: {str(e)}")
            # 确保资源被释放
            if hasattr(player_ref, "_actor_handle") and player_ref._actor_handle is not None:
                try:
                    player_ref.check_point()
                except:
                    pass

    end_time = time.time()
    log.info(f"Total Time Consumption: {end_time - begin_time}")

if __name__=='__main__':
    # 减少测试的Actor数量，避免资源耗尽
    ray.init(logging_level=logging.INFO)
    

    # # 将测试数量从500减少到50
    # for i in range(500):
    #     testPlayerActor(str(i))
    #     # time.sleep(0.1)  # 添加延迟以避免资源竞争

    # 使用asyncio实现异步执行
    import asyncio
    
    async def run_tests():
        tasks = []
        for i in range(500):
            # 将每个测试任务加入任务列表
            task = asyncio.create_task(asyncio.to_thread(testPlayerActor, str(i)))
            tasks.append(task)
        # 等待所有任务完成
        await asyncio.gather(*tasks)
    
    # 运行异步任务
    asyncio.run(run_tests())