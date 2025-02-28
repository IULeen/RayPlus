import sys

from core.actor import actorless_decorator

@actorless_decorator.actorless(name="Account", namespace="default")
class Account(object):

    def __init__(self, name="XiaoMing", CardID="123456", balance=0):
        self._name = name
        self._CardID = CardID
        self._balance = balance

    def deposit(self, amount):
        self._balance += amount
        return self._balance
    
    def withdraw(self, amount):
        self._balance -= amount
        return self._balance
    
    def get_info(self):
        return {"name": self._name, "CardID": self._CardID, "balance": self._balance}
    

    def _store_states(self):
        return self._name, self._CardID, self._balance
    
    def _recover_states(self, states):
        self._name, self._CardID, self._balance = states


# TODO 暂时先在模板函数中对 ray cluster 进行本地初始化，后续肯定需要考虑迁移到 ray cluster 上
# Account.bind(unique_id = "XiaoMing123456789", name="XiaoMing", CardID="123456789", balance=100)


