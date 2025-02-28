#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import time
import ray
import logging
import requests
import json

# 获取项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 添加这一行确保能访问到同级目录的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import logger
from core.actor.actorless_decorator import invoke

log = logger.get_logger(__name__)

from GamePlayerActor import GamePlayerActor

# HTTP API端点
RAYLESS_API_ENDPOINT = "http://localhost:5000/invoke"
RAYLESS_REGISTER_ENDPOINT = "http://localhost:5000/register"

def http_initialize_player_actors(total_players=50):
    """使用HTTP API初始化所有玩家Actor实例，确保它们在系统中注册"""
    log.info(f"开始通过HTTP API初始化 {total_players} 个玩家Actor实例...")
    
    start_time = time.time()
    
    # 初始化所有玩家实例
    for i in range(total_players):
        player_id = f"Player{i}"
        log.info(f"初始化玩家 {player_id}")
        
        # 注册Actor实例
        try:
            register_data = {
                "template_name": "GamePlayerActor",
                "actor_id": player_id,
                "init_args": [],
                "init_kwargs": {"player_id": player_id}
            }
            response = requests.post(RAYLESS_REGISTER_ENDPOINT, json=register_data)
            if response.status_code == 200:
                log.info(f"注册玩家 {player_id} 成功: {response.json()}")
            elif response.status_code == 400 and "已存在" in response.text:
                log.info(f"玩家 {player_id} 已存在，无需重新注册")
            else:
                log.warning(f"注册玩家 {player_id} 失败: {response.text}")
        except Exception as e:
            log.warning(f"注册玩家 {player_id} 时出错: {str(e)}")
        
        # 调用Actor初始化
        try:
            request_data = {
                "actor_id": player_id,
                "actor_type": "GamePlayerActor",
                "func_name": "get_info",
                "keep_alive": True,
                "params": []
            }
            
            response = requests.post(RAYLESS_API_ENDPOINT, json=request_data)
            
            if response.status_code == 200:
                result = response.json().get("result", {})
                log.info(f"玩家 {player_id} 初始化成功，状态: {result}")
            else:
                log.error(f"初始化玩家 {player_id} 失败: {response.text}")
        except Exception as e:
            log.error(f"初始化玩家 {player_id} 时出错: {str(e)}")
    
    end_time = time.time()
    log.info(f"所有 {total_players} 个玩家实例初始化完成，耗时: {end_time - start_time:.2f} 秒")

def initialize_player_actors(total_players=50):
    """使用Ray API初始化所有玩家Actor实例，确保它们在系统中注册"""
    log.info(f"开始初始化 {total_players} 个玩家Actor实例...")
    
    start_time = time.time()
    player_refs = []
    
    # 初始化所有玩家实例
    for i in range(total_players):
        player_id = f"Player{i}"
        log.info(f"初始化玩家 {player_id}")
        
        # 创建Player引用并注册
        player_ref = GamePlayerActor.bind(unique_id=player_id, player_id=player_id)
        player_ref.options(unique_id=player_id, is_persistent=True, storage_level=(0, 3))
        player_refs.append(player_ref)
        
        # 执行一次调用以确保Actor被创建
        result = invoke(player_ref, True, "get_info")
        log.info(f"玩家 {player_id} 初始化成功，状态: {result}")
    
    end_time = time.time()
    log.info(f"所有 {total_players} 个玩家实例初始化完成，耗时: {end_time - start_time:.2f} 秒")
    
    return player_refs

if __name__ == "__main__":
    # 检查HTTP服务器是否可用
    try:
        response = requests.get("http://localhost:5000/health")
        if response.status_code == 200:
            log.info("HTTP服务器已运行，使用HTTP API初始化玩家Actor")
            # 使用HTTP API初始化玩家Actor
            http_initialize_player_actors(50)
        else:
            log.warning("HTTP服务器返回异常状态码，将使用直接Ray API初始化玩家Actor")
            # 初始化Ray
            ray.init(logging_level=logging.INFO)
            # 使用Ray API初始化玩家Actor
            initialize_player_actors(50)
    except requests.ConnectionError:
        log.warning("无法连接到HTTP服务器，将使用直接Ray API初始化玩家Actor")
        # 初始化Ray
        ray.init(logging_level=logging.INFO)
        # 使用Ray API初始化玩家Actor
        initialize_player_actors(50)
    
    try:
        # 这里可以添加额外的验证步骤
        log.info("等待2秒以确保所有状态都已持久化...")
        time.sleep(2)
        
        # 验证所有玩家都可以访问
        for i in range(5):  # 仅验证前5个玩家以节省时间
            player_id = f"Player{i}"
            try:
                # 使用HTTP API验证
                request_data = {
                    "actor_id": player_id,
                    "actor_type": "GamePlayerActor",
                    "func_name": "get_info",
                    "keep_alive": False,
                    "params": []
                }
                
                response = requests.post(RAYLESS_API_ENDPOINT, json=request_data)
                
                if response.status_code == 200:
                    result = response.json().get("result", {})
                    log.info(f"验证玩家 {player_id} 成功，状态: {result}")
                else:
                    log.error(f"验证玩家 {player_id} 失败: {response.text}")
            except Exception as e:
                log.error(f"验证玩家 {player_id} 失败: {str(e)}")
        
        log.info("初始化脚本执行完成，现在可以运行压力测试了")
        
    except Exception as e:
        log.error(f"初始化过程中发生错误: {str(e)}")
    finally:
        # 不要关闭Ray，因为压力测试需要使用同一个Ray集群
        pass 