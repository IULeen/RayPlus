# 启动 Ray 集群前，先设置 Actor Template Package 的 PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/core/template
export PYTHONPATH=$PYTHONPATH:$(pwd)/tests

# 启动 Ray 集群
ray start --head --port=6379 --dashboard-port=8265 --dashboard-host=0.0.0.0

# 启动 FastAPI 服务
python core/transport/http_server.py

# 注意，在实验结束后，需要关闭 Ray 集群
# ray stop
