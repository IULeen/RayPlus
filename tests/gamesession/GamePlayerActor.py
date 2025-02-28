from core.actor import actorless_decorator

import json
import zlib
import time
from typing import Dict
from dataclasses import dataclass
from utils import logger
import sys

log = logger.get_logger(__name__)

@actorless_decorator.actorless(name="GamePlayerActor", namespace="default")
class GamePlayerActor(object):
    """在线游戏玩家状态Actor模板"""
    
    def __init__(self, player_id: str):
        log.info(f"Initializing GamePlayerActor with player_id: {player_id}")
        self.player_id = player_id
        
        # 基础状态属性
        self.position = {"x":0.0, "y":0.0, "z":0.0}
        self.velocity = {"x":0.0, "y":0.0, "z":0.0}
        self.health = 100.0
        self.skills = {}
        
        # 生成约1MB数据
        self._generate_initial_data()
        
        self.last_active = time.time()
        print(f"######################Player state size: {self._calculate_state_size()/1024/1024:.2f} MB")
        
    def _generate_initial_data(self):
        """生成初始状态数据（约1MB）"""
        # 装备数据（约0.8MB）
        self.inventory = [{
            "item_id": f"item_{i:06d}",  # 增加ID长度
            "durability": 100.0,
            "enhancement": i % 5,
            "modifiers": ["fire", "ice", "lightning"][:i%3+1]  # 添加更多数据
        } for i in range(3000)]  # 增加数量到12000
        
        # 动画数据（压缩前约1.5MB，压缩后约0.2MB）
        bone_matrices = [[float(j) for j in range(64)] for _ in range(3000)]  # 增大矩阵数据
        self.animation_state = zlib.compress(
            json.dumps({"bone_matrices": bone_matrices}).encode()
        )
        
    def _calculate_state_size(self):
        """递归计算实际内存占用"""
        def get_size(obj, seen=None):
            size = sys.getsizeof(obj)
            if seen is None:
                seen = set()
            obj_id = id(obj)
            if obj_id in seen:
                return 0
            seen.add(obj_id)
            
            if isinstance(obj, dict):
                size += sum(get_size(k, seen) + get_size(v, seen) for k, v in obj.items())
            elif isinstance(obj, (list, tuple, set)):
                size += sum(get_size(item, seen) for item in obj)
            return size
        
        total = 0
        for attr in ['inventory', 'animation_state', 'position', 'velocity', 'skills']:
            total += get_size(getattr(self, attr))
        return total

    def get_info(self):
        log.info(f"Getting info for player: {self.player_id}")
        return self.player_id, self.health
    
    def update_position(self, new_pos: Dict[str, float], new_vel: Dict[str, float]) -> bool:
        """更新玩家位置和速度"""
        self.position = new_pos
        self.velocity = new_vel
        self.last_active = time.time()
        return True
    
    def use_skill(self, skill_id: str) -> Dict:
        """使用技能并更新冷却状态"""
        skill = self.skills.get(skill_id, {"cooldown": 5.0, "level":1})
        # 此处省略冷却逻辑
        self.last_active = time.time()
        return skill

    
    def check_inactive(self, timeout: int = 300) -> bool:
        """检查玩家是否处于非活跃状态（触发状态降级）"""
        return (time.time() - self.last_active) > timeout

    def _store_states(self):
        """存储玩家状态（符合Rayless规范）"""
        log.info(f"正在存储玩家状态: {self.player_id}")
        # 使用zlib压缩和json序列化减小状态大小
        compressed_data = json.dumps({
                'player_id': self.player_id,
                'position': self.position,
                'health': self.health,
                'inventory': self.inventory[:],
                'last_active': self.last_active
            }).encode()

        
        # 封装为Rayless期望的格式
        state_dict = {
            'player_id': self.player_id,
            'compressed_data': compressed_data,
            'compression_type': 'zlib+json',
            'timestamp': time.time(),
            'sizeofState': sys.getsizeof(self._calculate_state_size())
        }
        
        log.info(f"状态存储完成，压缩后大小: {sys.getsizeof(compressed_data)}字节，原始状态大小: {sys.getsizeof(self._calculate_state_size())}字节")
        return state_dict
    
    def _recover_states(self, states):
        """从存储的状态恢复玩家状态（符合Rayless规范）"""
        log.info(f"正在恢复玩家状态: {states['player_id']}")
        
        if 'compressed_data' in states and states.get('compression_type') == 'zlib+json':
            # 从压缩数据恢复
            try:
                data = json.loads(states['compressed_data'].decode())
                
                # 恢复玩家基本信息
                self.player_id = data['player_id']
                
                # 恢复各个状态属性
                self.position = data['position']
                self.health = data['health']
                self.inventory = data['inventory']
                self.last_active = data.get('last_active', time.time())
                
                log.info(f"玩家 {self.player_id} 状态恢复成功")
            except Exception as e:
                log.error(f"状态恢复失败: {str(e)}")
                # 如果恢复失败，保持当前状态
        else:
            # 兼容旧格式数据
            log.info(f"使用旧格式恢复玩家状态")
            self.player_id = states['player_id']
            if 'state' in states:
                self.position = states['state']['position']
                self.velocity = states['state']['velocity']
                self.health = states['state']['health']
                self.skills = states['state']['skills']
                self.inventory = states['state']['inventory']
                self.animation_state = states['state']['animation_state']
                
        log.info(f"玩家 {self.player_id} 状态已恢复")