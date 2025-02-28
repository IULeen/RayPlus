#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Locust配置文件，用于Rayless性能测试
使用方法:
    1. 确保HTTP服务器已运行: python core/transport/http_server.py
    2. 先运行初始化脚本: python rayless_init_test.py
    3. 使用均匀分布测试: locust -f locustfile.py UniformRaylessUser --host http://localhost:5000
    4. 使用Zipfian分布测试: locust -f locustfile.py ZipfianRaylessUser --host http://localhost:5000
    5. 运行数据分析脚本: python rayless_plot_results.py

HTTP API调用格式:
    curl -X POST http://localhost:5000/invoke \\
    -H "Content-Type: application/json" \\
    -d '{
        "actor_id": "Player9",
        "actor_type": "GamePlayerActor",
        "func_name": "get_info",
        "keep_alive": true
    }'
"""

from rayless_locust_test import UniformRaylessUser, ZipfianRaylessUser