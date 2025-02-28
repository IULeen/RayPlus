#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import time
import random
import numpy as np
import json
import requests
from locust import User, task, between, events, HttpUser
from typing import Dict, List, Any

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
# 添加这一行确保能访问到同级目录的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import logger

log = logger.get_logger(__name__)

# 配置参数
TOTAL_PLAYERS = 500  # Player0~Player49
TEST_DURATION = 60  # 测试持续时间（秒）
RAYLESS_API_ENDPOINT = "http://localhost:5000/invoke"  # HTTP API端点

# 用于记录数据的全局变量
request_latencies = []
request_throughputs = []
distribution_type = "uniform"  # 可以是 "uniform" 或 "zipfian"

# 事件钩子，在测试开始时记录开始时间
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    global test_start_time, request_throughputs
    test_start_time = time.time()
    # 初始化吞吐量数组
    request_throughputs = [0] * TEST_DURATION
    log.info(f"性能测试开始，分布类型: {distribution_type}")

# 事件钩子，在测试结束时生成性能数据图表
@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    log.info("性能测试结束，正在生成性能数据图表...")
    save_performance_data()
    generate_performance_plots()

def save_performance_data():
    """将性能数据保存到文件"""
    data = {
        "distribution_type": distribution_type,
        "request_latencies": request_latencies,
        "request_throughputs": request_throughputs
    }
    
    filename = f"Ray+_performance_{distribution_type}_{int(time.time())}.json"
    with open(filename, "w") as f:
        json.dump(data, f)
    
    log.info(f"性能数据已保存到文件: {filename}")

def generate_performance_plots():
    """生成性能数据图表"""
    try:
        # 这个函数将在另一个脚本中实现
        # 它会从保存的JSON文件中读取数据并生成图表
        log.info("请运行 rayless_plot_results.py 生成图表")
    except Exception as e:
        log.error(f"生成图表时出错: {str(e)}")

# Zipfian分布生成器
class ZipfianGenerator:
    def __init__(self, n, skew=1.0):
        """初始化Zipfian分布生成器
        
        Args:
            n: 物品总数（从0到n-1）
            skew: 分布偏斜参数，越大偏斜越明显
        """
        self.n = n
        self.skew = skew
        
        # 预计算概率分布
        self.dist = np.power(np.arange(1, n+1), -skew)
        self.dist /= np.sum(self.dist)
        self.dist_cumsum = np.cumsum(self.dist)
    
    def next_item(self):
        """生成下一个符合Zipfian分布的随机数"""
        # 使用二分查找确定随机位置
        u = random.random()
        idx = np.searchsorted(self.dist_cumsum, u)
        return idx

# Locust用户类 - 均匀分布
class UniformRaylessUser(HttpUser):
    wait_time = between(0.1, 0.5)  # 思考时间
    host = "http://localhost:5000"  # Rayless HTTP服务器地址
    
    def on_start(self):
        """用户开始测试时的初始化操作"""
        global distribution_type
        distribution_type = "uniform"
        log.info("使用均匀分布负载")
    
    @task
    def invoke_player_methods(self):
        # 均匀分布随机选择玩家ID
        player_id = f"Player{random.randint(0, TOTAL_PLAYERS-1)}"
        
        # 随机选择一个方法
        methods = ["get_info", "update_position", "use_skill", "check_inactive"]
        method_weights = [0.4, 0.3, 0.2, 0.1]  # 各方法的调用概率权重
        method = random.choices(methods, weights=method_weights)[0]
        
        # 准备方法参数
        params = {}
        
        if method == "update_position":
            # 为update_position方法准备参数
            new_pos = {"x": random.uniform(-100, 100), "y": random.uniform(-100, 100), "z": random.uniform(-100, 100)}
            new_vel = {"x": random.uniform(-10, 10), "y": random.uniform(-10, 10), "z": random.uniform(-10, 10)}
            params = [new_pos, new_vel]
        elif method == "use_skill":
            # 为use_skill方法准备参数
            skill_id = f"skill_{random.randint(1, 10)}"
            params = [skill_id]
        elif method == "check_inactive":
            # 为check_inactive方法准备参数
            timeout = random.randint(100, 600)
            params = [timeout]
        
        # 准备HTTP请求数据
        # keep_alive = random.random() < 0.2  # 20%的概率保持Actor alive
        keep_alive = True
        
        request_data = {
            "actor_id": player_id,
            "actor_type": "GamePlayerActor",
            "func_name": method,
            "keep_alive": keep_alive,
            "params": params
        }
        
        # 记录请求开始时间
        start_time = time.time()
        
        try:
            # 发送HTTP请求调用Rayless Actor方法
            response = self.client.post("/invoke", json=request_data)
            
            # 记录端到端延迟
            latency = (time.time() - start_time) * 1000  # 转换为毫秒
            
            # 检查响应状态
            if response.status_code == 200:
                result = response.json().get("result")
                
                # 记录成功请求的数据
                request_latencies.append({
                    "timestamp": time.time() - test_start_time,
                    "player_id": player_id,
                    "method": method,
                    "latency": latency,
                    "keep_alive": keep_alive,
                    "success": True
                })
                
                # 更新吞吐量统计
                current_time = int(time.time() - test_start_time)
                if current_time < len(request_throughputs):
                    request_throughputs[current_time] += 1
                else:
                    # 填充可能缺失的时间段
                    while len(request_throughputs) <= current_time:
                        request_throughputs.append(0)
                    request_throughputs[current_time] += 1
                
                # 请求成功 - 使用新的API
                self.environment.events.request.fire(
                    request_type="invoke",
                    name=f"{player_id}.{method}",
                    response_time=latency,
                    response_length=len(response.text),
                    exception=None,
                    context={},
                    url="/invoke"
                )
                
            else:
                # 处理错误响应
                error_msg = f"HTTP错误: {response.status_code}, 响应: {response.text}"
                raise Exception(error_msg)
                
        except Exception as e:
            # 记录端到端延迟（即使失败）
            latency = (time.time() - start_time) * 1000  # 转换为毫秒
            request_latencies.append({
                "timestamp": time.time() - test_start_time,
                "player_id": player_id,
                "method": method,
                "latency": latency,
                "keep_alive": keep_alive,
                "success": False,
                "error": str(e)
            })
            
            # 请求失败 - 使用新的API
            self.environment.events.request.fire(
                request_type="invoke",
                name=f"{player_id}.{method}",
                response_time=latency,
                response_length=0,
                exception=e,
                context={},
                url="/invoke"
            )
            log.error(f"调用失败: {player_id}.{method}: {str(e)}")

# Locust用户类 - Zipfian分布
class ZipfianRaylessUser(HttpUser):
    wait_time = between(0.1, 0.5)  # 思考时间
    host = "http://localhost:5000"  # Rayless HTTP服务器地址
    
    def on_start(self):
        """用户开始测试时的初始化操作"""
        global distribution_type
        distribution_type = "zipfian"
        self.zipf_gen = ZipfianGenerator(TOTAL_PLAYERS, skew=1.2)  # 初始化Zipfian分布生成器
        log.info("使用Zipfian分布负载")
    
    @task
    def invoke_player_methods(self):
        # 使用Zipfian分布选择玩家ID
        player_idx = self.zipf_gen.next_item()
        player_id = f"Player{player_idx}"
        
        # 随机选择一个方法
        methods = ["get_info", "update_position", "use_skill", "check_inactive"]
        method_weights = [0.4, 0.3, 0.2, 0.1]  # 各方法的调用概率权重
        method = random.choices(methods, weights=method_weights)[0]
        
        # 准备方法参数
        params = {}
        
        if method == "update_position":
            # 为update_position方法准备参数
            new_pos = {"x": random.uniform(-100, 100), "y": random.uniform(-100, 100), "z": random.uniform(-100, 100)}
            new_vel = {"x": random.uniform(-10, 10), "y": random.uniform(-10, 10), "z": random.uniform(-10, 10)}
            params = [new_pos, new_vel]
        elif method == "use_skill":
            # 为use_skill方法准备参数
            skill_id = f"skill_{random.randint(1, 10)}"
            params = [skill_id]
        elif method == "check_inactive":
            # 为check_inactive方法准备参数
            timeout = random.randint(100, 600)
            params = [timeout]
        
        # 准备HTTP请求数据
        # keep_alive = random.random() < 0.2  # 20%的概率保持Actor alive
        keep_alive = True
        request_data = {
            "actor_id": player_id,
            "actor_type": "GamePlayerActor",
            "func_name": method,
            "keep_alive": keep_alive,
            "params": params
        }
        
        # 记录请求开始时间
        start_time = time.time()
        
        try:
            # 发送HTTP请求调用Rayless Actor方法
            response = self.client.post("/invoke", json=request_data)
            
            # 记录端到端延迟
            latency = (time.time() - start_time) * 1000  # 转换为毫秒
            
            # 检查响应状态
            if response.status_code == 200:
                result = response.json().get("result")
                
                # 记录成功请求的数据
                request_latencies.append({
                    "timestamp": time.time() - test_start_time,
                    "player_id": player_id,
                    "method": method,
                    "latency": latency,
                    "keep_alive": keep_alive,
                    "success": True
                })
                
                # 更新吞吐量统计
                current_time = int(time.time() - test_start_time)
                if current_time < len(request_throughputs):
                    request_throughputs[current_time] += 1
                else:
                    # 填充可能缺失的时间段
                    while len(request_throughputs) <= current_time:
                        request_throughputs.append(0)
                    request_throughputs[current_time] += 1
                
                # 请求成功 - 使用新的API
                self.environment.events.request.fire(
                    request_type="invoke",
                    name=f"{player_id}.{method}",
                    response_time=latency,
                    response_length=len(response.text),
                    exception=None,
                    context={},
                    url="/invoke"
                )
                
            else:
                # 处理错误响应
                error_msg = f"HTTP错误: {response.status_code}, 响应: {response.text}"
                raise Exception(error_msg)
                
        except Exception as e:
            # 记录端到端延迟（即使失败）
            latency = (time.time() - start_time) * 1000  # 转换为毫秒
            request_latencies.append({
                "timestamp": time.time() - test_start_time,
                "player_id": player_id,
                "method": method,
                "latency": latency,
                "keep_alive": keep_alive,
                "success": False,
                "error": str(e)
            })
            
            # 请求失败 - 使用新的API
            self.environment.events.request.fire(
                request_type="invoke",
                name=f"{player_id}.{method}",
                response_time=latency,
                response_length=0,
                exception=e,
                context={},
                url="/invoke"
            )
            log.error(f"调用失败: {player_id}.{method}: {str(e)}") 