from core.actor import actorless_decorator

import json
import zlib
import time
import numpy as np
import sys
import threading
from typing import Dict, List, Tuple, Any
from utils import logger

log = logger.get_logger(__name__)

@actorless_decorator.actorless(name="PSActor", namespace="default")
class PSActor(object):
    """参数服务器Actor，用于分布式训练中存储和更新模型参数"""
    
    def __init__(self, model_id: str, param_shape: Dict[str, List[int]] = None):
        """
        初始化参数服务器Actor
        
        参数:
            model_id: 模型唯一标识符
            param_shape: 参数形状字典，格式为 {参数名: 形状列表}
        """
        log.info(f"初始化参数服务器 PSActor，模型ID: {model_id}")
        self.model_id = model_id
        self.params = {}
        self.param_version = 0
        self.update_lock = threading.Lock()
        self.last_update_time = time.time()
        self.update_count = 0
        self.grad_stats = {}  # 梯度统计信息
        
        # 初始化参数（如果提供了形状）
        if param_shape:
            self._initialize_params(param_shape)
            log.info(f"参数初始化完成，总大小: {self._calculate_params_size()/1024/1024:.2f} MB")
        
        # 性能统计
        self.performance_stats = {
            "total_updates": 0,
            "total_pulls": 0,
            "update_time_ms": [],
            "pull_time_ms": [],
            "start_time": time.time()
        }
    
    def _initialize_params(self, param_shape: Dict[str, List[int]]):
        """初始化模型参数"""
        for name, shape in param_shape.items():
            # 使用Xavier初始化
            fan_in = shape[0] if len(shape) >= 1 else 1
            fan_out = shape[1] if len(shape) >= 2 else 1
            limit = np.sqrt(6 / (fan_in + fan_out))
            self.params[name] = np.random.uniform(-limit, limit, size=shape).astype(np.float32)
            
            # 初始化梯度统计
            self.grad_stats[name] = {
                "updates": 0,
                "mean": 0,
                "variance": 0,
                "min": 0,
                "max": 0
            }
    
    def _calculate_params_size(self) -> int:
        """计算参数占用的内存大小（字节）"""
        total_size = 0
        for name, param in self.params.items():
            total_size += param.nbytes
        return total_size
    
    def get_params(self) -> Tuple[Dict[str, np.ndarray], int]:
        """
        获取当前参数和版本号
        
        返回:
            (params, version): 参数字典和当前版本号
        """
        start_time = time.time()
        
        # 创建参数的深拷贝以避免并发修改问题
        with self.update_lock:
            params_copy = {k: v.copy() for k, v in self.params.items()}
            version = self.param_version
        
        # 更新统计信息
        pull_time = (time.time() - start_time) * 1000  # 毫秒
        self.performance_stats["total_pulls"] += 1
        self.performance_stats["pull_time_ms"].append(pull_time)
        
        log.info(f"参数拉取完成，版本: {version}，耗时: {pull_time:.2f}ms")
        return params_copy, version
    
    def update_params(self, gradients: Dict[str, np.ndarray], learning_rate: float = 0.01) -> int:
        """
        使用梯度更新参数
        
        参数:
            gradients: 参数梯度字典
            learning_rate: 学习率
            
        返回:
            新版本号
        """
        start_time = time.time()
        
        with self.update_lock:
            # 应用梯度更新
            for name, grad in gradients.items():
                if name not in self.params:
                    log.warning(f"参数 {name} 不存在，跳过更新")
                    continue
                
                # 更新梯度统计
                self._update_grad_stats(name, grad)
                
                # 梯度下降更新
                self.params[name] -= learning_rate * grad
            
            # 更新版本和时间
            self.param_version += 1
            self.last_update_time = time.time()
            self.update_count += 1
        
        # 更新性能统计
        update_time = (time.time() - start_time) * 1000  # 毫秒
        self.performance_stats["total_updates"] += 1
        self.performance_stats["update_time_ms"].append(update_time)
        
        log.info(f"参数更新完成，新版本: {self.param_version}，耗时: {update_time:.2f}ms")
        return self.param_version
    
    def _update_grad_stats(self, name: str, grad: np.ndarray):
        """更新梯度统计信息"""
        stats = self.grad_stats[name]
        stats["updates"] += 1
        
        # 计算统计量
        grad_abs = np.abs(grad)
        new_min = np.min(grad)
        new_max = np.max(grad)
        new_mean = np.mean(grad_abs)
        
        # 更新统计信息
        n = stats["updates"]
        if n == 1:
            stats["mean"] = new_mean
            stats["min"] = new_min
            stats["max"] = new_max
            stats["variance"] = 0
        else:
            # 增量更新均值和方差
            old_mean = stats["mean"]
            stats["mean"] += (new_mean - old_mean) / n
            stats["variance"] += (new_mean - old_mean) * (new_mean - stats["mean"])
            stats["min"] = min(stats["min"], new_min)
            stats["max"] = max(stats["max"], new_max)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取参数服务器统计信息"""
        uptime = time.time() - self.performance_stats["start_time"]
        
        # 计算更新和拉取操作的平均时间
        avg_update_time = np.mean(self.performance_stats["update_time_ms"]) if self.performance_stats["update_time_ms"] else 0
        avg_pull_time = np.mean(self.performance_stats["pull_time_ms"]) if self.performance_stats["pull_time_ms"] else 0
        
        # 计算吞吐量（每秒操作数）
        updates_per_sec = self.performance_stats["total_updates"] / uptime if uptime > 0 else 0
        pulls_per_sec = self.performance_stats["total_pulls"] / uptime if uptime > 0 else 0
        
        return {
            "model_id": self.model_id,
            "param_version": self.param_version,
            "param_count": len(self.params),
            "param_size_mb": self._calculate_params_size() / (1024 * 1024),
            "update_count": self.update_count,
            "last_update": self.last_update_time,
            "uptime_seconds": uptime,
            "performance": {
                "total_updates": self.performance_stats["total_updates"],
                "total_pulls": self.performance_stats["total_pulls"],
                "avg_update_time_ms": avg_update_time,
                "avg_pull_time_ms": avg_pull_time,
                "updates_per_second": updates_per_sec,
                "pulls_per_second": pulls_per_sec
            },
            "grad_stats": {name: stats for name, stats in self.grad_stats.items()}
        }
    
    def run_performance_test(self, num_iterations: int = 100, param_size: int = 1000000) -> Dict[str, Any]:
        """
        运行简单的性能测试
        
        参数:
            num_iterations: 测试迭代次数
            param_size: 测试参数大小（元素个数）
            
        返回:
            测试结果统计
        """
        log.info(f"开始性能测试: {num_iterations}次迭代，参数大小: {param_size}")
        
        # 初始化测试参数
        test_shape = {"weight": [1000, param_size // 1000]}
        self._initialize_params(test_shape)
        
        # 创建测试梯度
        test_grad = {"weight": np.random.normal(0, 0.01, test_shape["weight"]).astype(np.float32)}
        
        # 重置性能统计
        self.performance_stats = {
            "total_updates": 0,
            "total_pulls": 0,
            "update_time_ms": [],
            "pull_time_ms": [],
            "start_time": time.time()
        }
        
        # 运行测试
        for i in range(num_iterations):
            if i % 10 == 0:
                log.info(f"性能测试进度: {i}/{num_iterations}")
            
            # 拉取参数
            self.get_params()
            
            # 更新参数
            self.update_params(test_grad)
        
        # 返回测试结果
        return self.get_stats()
    
    def _store_states(self) -> Dict:
        """存储参数服务器状态（符合Rayless规范）"""
        log.info(f"正在存储参数服务器状态: {self.model_id}")
        
        # 将numpy数组转换为列表以便JSON序列化
        serializable_params = {k: v.tolist() for k, v in self.params.items()}
        
        # 准备状态数据
        state_data = {
            'model_id': self.model_id,
            'param_version': self.param_version,
            'params': serializable_params,
            'update_count': self.update_count,
            'last_update_time': self.last_update_time,
            'performance_stats': self.performance_stats,
            'timestamp': time.time()
        }
        
        # 压缩状态数据
        compressed_data = zlib.compress(json.dumps(state_data).encode())
        
        # 封装为Rayless期望的格式
        state_dict = {
            'model_id': self.model_id,
            'compressed_data': compressed_data,
            'compression_type': 'zlib+json',
            'timestamp': time.time(),
            'param_version': self.param_version,
            'sizeofState': sys.getsizeof(compressed_data)
        }
        
        log.info(f"状态存储完成，压缩后大小: {sys.getsizeof(compressed_data)/1024:.2f} KB")
        return state_dict
    
    def _recover_states(self, states: Dict):
        """从存储的状态恢复参数服务器状态（符合Rayless规范）"""
        log.info(f"正在恢复参数服务器状态: {states['model_id']}")
        
        if 'compressed_data' in states and states.get('compression_type') == 'zlib+json':
            try:
                # 解压缩数据
                decompressed_data = zlib.decompress(states['compressed_data'])
                data = json.loads(decompressed_data.decode())
                
                # 恢复基本信息
                self.model_id = data['model_id']
                self.param_version = data['param_version']
                self.update_count = data['update_count']
                self.last_update_time = data['last_update_time']
                
                # 恢复性能统计
                self.performance_stats = data['performance_stats']
                
                # 恢复参数（转回numpy数组）
                self.params = {k: np.array(v, dtype=np.float32) for k, v in data['params'].items()}
                
                # 初始化梯度统计
                self.grad_stats = {name: {
                    "updates": 0,
                    "mean": 0,
                    "variance": 0,
                    "min": 0,
                    "max": 0
                } for name in self.params}
                
                log.info(f"参数服务器 {self.model_id} 状态恢复成功，参数版本: {self.param_version}")
            except Exception as e:
                log.error(f"状态恢复失败: {str(e)}")
        else:
            log.error(f"不支持的状态格式或缺少压缩数据")
