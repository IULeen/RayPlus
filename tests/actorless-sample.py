import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from utils import logger
from core.actor import actorless_decorator
from core.actor.actorless_decorator import invoke, invoke_by_actorlessID

log = logger.get_logger(__name__)


@actorless_decorator.actorless(name="Counter", namespace="default")
class Counter(object):
    def __init__(self, name, init_value=0):
        log.info("Actor is inited!")
        self._name = name
        self.state = init_value

    def get_state(self):
        log.info("-------------I am invoked!--------------")
        return self._name, self.state
    
    def set_state(self, value):
        self.state = value

    def add_state(self, value):
        self.state += value

    # 目前需要用户在定义Actor模板函数时，实现 _store_states 和 _recover_states 函数
    def _store_states(self):
        return self._name, self.state
    
    def _recover_states(self, states):
        self._name, self.state = states


def testCounterActor():
    state_id = "XiaoHong"
    name = state_id

    actorless_Ref_1 = Counter.bind(state_id, name=name, init_value=1)
    actorless_Ref_1.options(is_persistent=True, storage_level=(2, 2))
    log.info(f"-------------------------------------------")
    keep_alive = True
    res = invoke(actorless_Ref_1, keep_alive, "get_state", )
    log.info(f"1. res: {res}") 
    res = invoke(actorless_Ref_1, keep_alive, "set_state", 100)
    log.info(f"2. res: {res}") 
    res = invoke(actorless_Ref_1, keep_alive, "get_state", ) 
    log.info(f"3. res: {res}")  
    res = invoke_by_actorlessID("Counter", state_id, keep_alive, "get_state", )
    log.info(f"4. res: {res}")

    # invoke(actorless_Ref_1, keep_alive, "add_state", 200)
    # invoke(actorless_Ref_1, keep_alive, "get_state", )



from Account import Account
def testAccountActor():

    name = "XiaoMing"
    CardID = "123456789"
    balance = 0

    keep_alive = True
    account_ref = Account.bind(unique_id = "XiaoMing123456789", name=name, CardID=CardID, balance=balance)
    account_ref.options(is_persistent=True, storage_level=(2, 2))
    

    invocation_start_time = time.time()
    res = invoke(account_ref, keep_alive, "deposit", 500)
    invocation_end_time = time.time()
    log.info(f"Time Consumption: {invocation_end_time - invocation_start_time}, Balance: {res}")


    invocation_start_time = time.time()
    res = invoke(account_ref, keep_alive, "deposit", 400)
    invocation_end_time = time.time()
    log.info(f"Time Consumption: {invocation_end_time - invocation_start_time}, Balance: {res}")

    invocation_start_time = time.time()
    res = invoke(account_ref, keep_alive, "deposit", 300)
    invocation_end_time = time.time()
    log.info(f"Time Consumption: {invocation_end_time - invocation_start_time}, Balance: {res}")

    invocation_start_time = time.time()
    res = invoke(account_ref, keep_alive, "withdraw", 100)
    invocation_end_time = time.time()
    log.info(f"Time Consumption: {invocation_end_time - invocation_start_time}, Balance: {res}")

    invocation_start_time = time.time()
    res = invoke(account_ref, keep_alive, "get_info", )
    invocation_end_time = time.time()
    log.info(f"Time Consumption: {invocation_end_time - invocation_start_time}, Info: {res}")

    # 添加一个根据 actorless_id 调用函数的测试，这个 unique_id 按道理应该由谁来生成呢？
    # 暂时就先由 调用方自己指定吧，这样的话， bind 函数对应是否需要添加 unique_id 是否冲突的检查？
    invocation_start_time = time.time()
    account_id = name + CardID
    res = invoke_by_actorlessID("Account", account_id, keep_alive, "get_info", )
    invocation_end_time = time.time()
    log.info(f"Time Consumption: {invocation_end_time - invocation_start_time}, Info: {res}")

    invocation_start_time = time.time()
    account_id = name + CardID
    res = invoke_by_actorlessID("Account", account_id, keep_alive, "withdraw", 200)
    invocation_end_time = time.time()
    log.info(f"Time Consumption: {invocation_end_time - invocation_start_time}, Balance: {res}")


    # 测试从 Account Actor Orchestrator 中调用 Counter Actor 的 get_state 方法
    # 这个测试的目的是测试 Actorless 的跨 Actor 调用
    res = invoke_by_actorlessID("Counter", "XiaoHong", keep_alive, "get_state", )
    log.info(f"Invoke from Account Actor, res: {res}")

    # res = invoke_by_actorlessID("Counter", "XiaoHong", keep_alive, "set_state", 1000)
    # log.info(f"Invoke from Account Actor, res: {resHong
    # res = invoke_by_actorlessID("Counter", "XiaoHong", keep_alive, "get_state", )
    # log.info(f"Invoke from Account Actor, res: {res}")


if __name__=='__main__':
    testCounterActor()
    testAccountActor()

