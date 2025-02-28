import sys
import os
import ray
import logging
import importlib.util
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../tests"))  # 添加测试目录

from fastapi import FastAPI, HTTPException
from core.actor.actorless_decorator import invoke_by_http, deleteActorless
from core.service.actorless_lookup_service import ActorlessLookupService
from utils import logger
from utils.annotations import Deprecated
from config.setting import RAY_INIT_CONFIG

# 定义请求和响应模型
class InvokeRequest(BaseModel):
    actor_id: str
    actor_type: str
    func_name: str
    keep_alive: bool = False
    params: Union[List[Any], Dict[str, Any]] = []

class RegisterRequest(BaseModel):
    template_name: str
    actor_id: str
    init_args: List[Any] = []
    init_kwargs: Dict[str, Any] = {}

class DeleteRequest(BaseModel):
    actor_id: str
    actor_type: str

class ActorInfo(BaseModel):
    actor_id: str
    actor_type: str
    is_alive: bool
    namespace: str
    state: Optional[Dict[str, Any]] = None

app = FastAPI(title="Actorless API", description="Actor 实例管理与调用服务")
lookup_service = ActorlessLookupService()
log = logger.get_logger(__name__)

@Deprecated
def original_init_ray():
    """初始化 Ray 实例或连接到已有集群"""
    if not ray.is_initialized():
        try:
            # 尝试连接到已有集群

            ray.init(
                address="localhost:6379",  # 自动连接到已有集群
                ignore_reinit_error=True,
                logging_level=logging.INFO,
                # **RAY_INIT_CONFIG
            )
            log.info("成功连接到 Ray 集群")
        except Exception as e:
            log.warning(f"连接集群失败，将创建本地 Ray 实例: {str(e)}")
            # 如果连接失败，则创建本地实例
            ray.init(
                ignore_reinit_error=True,
                logging_level=logging.INFO,
                # **RAY_INIT_CONFIG
            )

@Deprecated
def init_ray_local():
    """初始化本地 Ray 实例"""
    if not ray.is_initialized():
        try:
            # 设置正确的运行时环境
            runtime_env = {
                "py_modules": [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests")],
                "env_vars": {"PYTHONPATH": os.environ.get("PYTHONPATH", "")}
            }
            ray.init(
                ignore_reinit_error=True,
                logging_level=logging.INFO,
                runtime_env=runtime_env,
                **RAY_INIT_CONFIG
            )
        except Exception as e:
            log.warning(f"初始化本地 Ray 实例失败: {str(e)}")

# 修改建议：增强HTTP服务的Ray初始化
def init_ray():
    """更健壮的Ray初始化方法"""
    if not ray.is_initialized():
        try:
            # 确定项目根目录
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            log.info(f"################project_root: {project_root}")
            # 设置运行时环境
            runtime_env = {
                "working_dir": project_root,
                "env_vars": {"PYTHONPATH": f"{os.environ.get('PYTHONPATH', '')}:{project_root}",},
                "excludes": [f"{project_root}/trace/**"]
            }
            
            # 尝试连接现有集群
            ray.init(
                address="auto",  # 自动检测集群
                ignore_reinit_error=True,
                logging_level=logging.INFO,
                runtime_env=runtime_env
            )
            log.info("成功连接到Ray集群")
        except Exception as e:
            log.warning(f"连接集群失败，创建本地实例: {str(e)}")
            ray.init(
                ignore_reinit_error=True,
                logging_level=logging.INFO,
                runtime_env=runtime_env
            )

# 在应用启动时初始化
init_ray()
# init_ray_local()

def load_template(template_name: str):
    """动态加载 Actor 模板类"""
    root_dir = Path(__file__).parent.parent.parent
    template_path = root_dir / "templates" / f"{template_name}.py"
    
    if not template_path.exists():
        raise FileNotFoundError(f"模板文件不存在: {template_path}")
    
    try:
        spec = importlib.util.spec_from_file_location(template_name, template_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        template_class = getattr(module, template_name)
        return template_class
    except Exception as e:
        raise ImportError(f"加载模板失败: {str(e)}")

@app.get("/health")
async def health_check():
    """健康检查端点，用于确认服务器是否正常运行"""
    try:
        # 检查Ray是否已初始化
        ray_status = ray.is_initialized()
        # 返回服务状态
        return {
            "status": "ok",
            "ray_initialized": ray_status,
            "service": "Rayless HTTP API"
        }
    except Exception as e:
        log.error(f"健康检查失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器状态异常: {str(e)}")

@app.post("/invoke", response_model=Dict[str, Any])
async def invoke(request: InvokeRequest):
    """调用 Actor 实例的方法"""
    try:
        if isinstance(request.params, list):
            result = invoke_by_http(request.actor_type, request.actor_id, 
                                  request.keep_alive, request.func_name, *request.params)
        elif isinstance(request.params, dict):
            result = invoke_by_http(request.actor_type, request.actor_id, 
                                  request.keep_alive, request.func_name, **request.params)
        else:
            result = invoke_by_http(request.actor_type, request.actor_id, 
                                  request.keep_alive, request.func_name, request.params)
        return {"result": result}
    except Exception as e:
        log.error(f"调用失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete")
async def delete(request: DeleteRequest):
    """删除 Actor 实例"""
    try:
        success, actorless_ref = lookup_service.find_actorless(request.actor_type, request.actor_id)
        if success:
            deleteActorless(actorless_ref)
            return {"result": f"Actor {request.actor_id} 删除成功"}
        else:
            raise HTTPException(status_code=404, detail=f"未找到 Actor: {request.actor_id}")
    except Exception as e:
        log.error(f"删除失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/register")
async def register(request: RegisterRequest):
    """注册新的 Actor 实例"""
    try:
        # 检查是否已存在同名实例
        success, existing_ref = lookup_service.find_actorless(request.template_name, request.actor_id)
        if success:
            raise HTTPException(status_code=400, detail=f"Actor {request.actor_id} 已存在")
        
        # 加载模板类
        template_class = load_template(request.template_name)
        
        # 创建新实例
        if request.init_args and request.init_kwargs:
            actor_ref = template_class.bind(request.actor_id, *request.init_args, **request.init_kwargs)
        elif request.init_args:
            actor_ref = template_class.bind(request.actor_id, *request.init_args)
        elif request.init_kwargs:
            actor_ref = template_class.bind(request.actor_id, **request.init_kwargs)
        else:
            actor_ref = template_class.bind(request.actor_id)
        
        return {
            "result": "注册成功",
            "actor_id": request.actor_id,
            "template_name": request.template_name
        }
        
    except FileNotFoundError as e:
        log.error(f"模板不存在: {str(e)}")
        raise HTTPException(status_code=404, detail=f"模板不存在: {request.template_name}")
    except Exception as e:
        log.error(f"注册失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/templates", response_model=List[Dict[str, Any]])
async def list_templates():
    """获取所有可用的 Actor 模板"""
    try:
        templates_dir = Path(__file__).parent.parent.parent / "templates"
        templates = []
        
        for template_file in templates_dir.glob("*.py"):
            if template_file.stem.startswith("__"):
                continue
                
            try:
                spec = importlib.util.spec_from_file_location(
                    template_file.stem, 
                    template_file
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                template_class = getattr(module, template_file.stem)
                
                # 获取模板描述信息
                description = getattr(template_class, "__doc__", "").strip() or "无描述"
                methods = [
                    name for name, func in vars(template_class).items()
                    if callable(func) and not name.startswith("_")
                ]
                
                templates.append({
                    "name": template_file.stem,
                    "description": description,
                    "methods": methods
                })
            except Exception as e:
                log.warning(f"加载模板 {template_file.name} 失败: {e}")
                
        return templates
    except Exception as e:
        log.error(f"获取模板列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # 在应用启动时调用
    # ensure_module_imports()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000) 