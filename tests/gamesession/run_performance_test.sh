#!/bin/bash

# 在脚本开头添加项目根目录到 PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/../..

echo "======================================"
echo "Ray+ 性能测试自动化脚本"
echo "======================================"

# 确保工作目录正确
cd "$(dirname "$0")"

# 检查依赖是否已安装
echo "检查依赖..."
python -c "import locust, matplotlib, numpy, requests" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "缺少必要的Python依赖，正在安装..."
    pip install locust matplotlib numpy requests
fi

# 清理之前的测试结果
echo "清理之前的测试结果..."
rm -f Ray+_performance_*.json
rm -f Ray+_*.png

# 确保HTTP服务器正在运行
echo "步骤1: 确认HTTP服务器是否运行..."
curl -s http://localhost:5000/health > /dev/null
if [ $? -ne 0 ]; then
    echo "HTTP服务器似乎未运行。正在尝试启动服务器..."
    # 启动HTTP服务器作为后台进程
    cd ../
    python core/transport/http_server.py > http_server.log 2>&1 &
    HTTP_SERVER_PID=$!
    cd tests/
    echo "HTTP服务器已启动，PID: $HTTP_SERVER_PID"
    echo "等待5秒让服务器完全启动..."
    sleep 5
    
    # 再次检查服务器是否运行
    curl -s http://localhost:5000/health > /dev/null
    if [ $? -ne 0 ]; then
        echo "无法启动HTTP服务器，请手动检查并启动服务器后再运行此脚本。"
        exit 1
    fi
else
    echo "HTTP服务器已经在运行。"
fi

# # 初始化步骤可以手动进行，目前有BUG
# # 步骤2: 初始化Ray和测试环境
# echo "步骤2: 初始化Ray和玩家Actor实例..."
# python rayless_init_test.py

# # 检查初始化是否成功
# if [ $? -ne 0 ]; then
#     echo "初始化失败，请检查日志并修复问题后重试。"
#     exit 1
# fi

# 初始化步骤：先生成N个Player实例，然后激活所有Player实例
echo "步骤2.0: 先生成N个Player实例..."
python Gamesession_Orchestrator.py
wait
echo "N个Player实例已生成"

echo "步骤2.1: 初始化并激活所有Player实例..."
for i in {0..499}; do
    curl -X POST http://localhost:5000/invoke \
    -H "Content-Type: application/json" \
    -d '{
        "actor_id": "Player'$i'",
        "actor_type": "GamePlayerActor", 
        "func_name": "get_info",
        "keep_alive": true
    }' > /dev/null 2>&1 &
done

# 等待所有请求完成
echo "等待所有Player实例初始化完成..."
wait
echo "所有Player实例已激活"




# echo "初始化完成，等待5秒..."
# sleep 5
# 步骤4: 运行Zipfian分布测试
echo "步骤4: 运行Zipfian分布测试..."
echo "使用无头模式运行30秒测试，20个用户，以10用户/秒的速率启动..."
locust -f locustfile.py ZipfianRaylessUser --headless -u 20 -r 10 --run-time 30s --host http://localhost:5000

echo "Zipfian分布测试完成，等待5秒..."
sleep 5

# 步骤3: 运行均匀分布测试
echo "步骤3: 运行均匀分布测试..."
echo "使用无头模式运行30秒测试，20个用户，以10用户/秒的速率启动..."
locust -f locustfile.py UniformRaylessUser --headless -u 20 -r 10 --run-time 30s --host http://localhost:5000

echo "均匀分布测试完成，等待5秒..."
sleep 5


# 步骤5: 生成性能分析图表
echo "步骤5: 生成性能分析图表..."
python Ray+_plot_results.py

echo "======================================"
echo "测试完成！请查看生成的图表分析结果。"
echo "图表文件:"
ls -la Ray+_*.png
echo "======================================" 