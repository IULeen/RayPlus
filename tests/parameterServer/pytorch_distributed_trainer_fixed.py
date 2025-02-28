import ray
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import os
import sys
import argparse
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any, Optional

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.parameterServer.PSActor import PSActor
from core.actor.actorless_decorator import invoke, invoke_by_actorlessID

# 配置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelPlayerActor:
    """
    模型训练玩家Actor，负责一个独立模型的训练
    """
    def __init__(self, player_id: str, model_type: str, model_id: str, 
                 ps_instance, dataset_path: str = './data', batch_size: int = 32):
        """
        初始化模型训练玩家
        
        参数:
            player_id: 玩家ID
            model_type: 模型类型 (resnet18, alexnet, vgg16)
            model_id: 参数服务器中的模型ID
            ps_instance: 预先绑定的参数服务器实例
            dataset_path: 数据集路径
            batch_size: 批处理大小
        """
        self.player_id = player_id
        self.model_type = model_type
        self.model_id = model_id
        self.batch_size = batch_size
        
        # 存储预先绑定的参数服务器实例
        self.ps_instance = ps_instance
        
        # 初始化模型
        self.model = self._create_model(model_type)
        self.model.to(device)
        
        # 初始化数据加载器
        self.train_loader, self.val_loader = self._load_dataset(dataset_path, batch_size)
        
        # 初始化损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        
        # 训练状态
        self.epoch = 0
        self.global_step = 0
        self.best_accuracy = 0.0
        self.train_losses = []
        self.val_accuracies = []
        
        print(f"模型玩家 {player_id} 初始化完成，模型类型: {model_type}")
    
    def _create_model(self, model_type: str) -> nn.Module:
        """创建指定类型的模型，适配小尺寸输入"""
        if model_type == 'resnet18':
            # 修改 ResNet 以适应 CIFAR-10
            model = models.resnet18(pretrained=False)
            # 修改第一个卷积层以适应 32x32 输入
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            # 移除 maxpool 层，它会使小尺寸输入过度缩小
            model.maxpool = nn.Identity()
            return model
            
        elif model_type == 'alexnet':
            # 为 CIFAR-10 创建简化版 AlexNet
            return nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 192, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(256 * 4 * 4, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 10),
            )
            
        elif model_type == 'vgg16':
            # 修改 VGG 以适应 CIFAR-10
            model = models.vgg16(pretrained=False)
            # 修改分类器以适应较小的特征图
            model.classifier = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 10),
            )
            return model
            
        elif model_type == 'mobilenet_v2':
            # 修改 MobileNetV2 以适应 CIFAR-10
            model = models.mobilenet_v2(pretrained=False)
            # 修改第一个卷积层
            model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
            # 修改分类器
            model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(1280, 10),
            )
            return model
            
        elif model_type == 'densenet121':
            # 修改 DenseNet 以适应 CIFAR-10
            model = models.densenet121(pretrained=False)
            # 修改第一个卷积层
            model.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            # 移除 maxpool 层
            model.features.pool0 = nn.Identity()
            # 修改分类器
            model.classifier = nn.Linear(1024, 10)
            return model
            
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    def _load_dataset(self, dataset_path: str, batch_size: int) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """加载CIFAR-10数据集"""
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        # 如果没有真实数据集，创建模拟数据
        try:
            train_dataset = datasets.CIFAR10(
                root=dataset_path, train=True, download=True, transform=transform_train)
            val_dataset = datasets.CIFAR10(
                root=dataset_path, train=False, download=True, transform=transform_val)
        except:
            print(f"无法加载CIFAR-10数据集，使用模拟数据")
            # 创建模拟数据集
            train_dataset = torch.utils.data.TensorDataset(
                torch.randn(10000, 3, 32, 32), torch.randint(0, 10, (10000,)))
            val_dataset = torch.utils.data.TensorDataset(
                torch.randn(2000, 3, 32, 32), torch.randint(0, 10, (2000,)))
        
        # 将 num_workers 设置为 0 以避免多进程问题
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        return train_loader, val_loader
    
    def model_to_param_dict(self) -> Dict[str, np.ndarray]:
        """将模型参数转换为字典格式，用于参数服务器"""
        param_dict = {}
        for name, param in self.model.named_parameters():
            # 直接使用numpy数组，避免不必要的转换
            param_dict[name] = param.data.cpu().numpy()
        return param_dict
    
    def param_dict_to_model(self, param_dict: Dict[str, np.ndarray]) -> None:
        """从参数字典更新模型参数"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in param_dict:
                    # 直接使用numpy数组，减少转换开销
                    param.copy_(torch.from_numpy(param_dict[name]).to(device))
    
    def compute_gradients(self, inputs, targets) -> Tuple[Dict[str, np.ndarray], float]:
        """计算一个批次的梯度"""
        # 清除之前的梯度
        self.optimizer.zero_grad()
        
        # 前向传播
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        
        # 反向传播
        loss.backward()
        
        # 收集梯度
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.cpu().numpy()
        
        return gradients, loss.item()
    
    def train_batch(self, keep_alive: bool = True) -> Dict[str, Any]:
        """训练一个批次"""
        try:
            # 1. 从参数服务器获取最新参数
            start_time = time.time()
            params, version = invoke(self.ps_instance, True, "get_params")
            pull_time = (time.time() - start_time) * 1000
            
            # 2. 更新本地模型参数
            self.param_dict_to_model(params)
            
            # 3. 获取一个批次的数据
            try:
                inputs, targets = next(iter(self.train_loader))
                inputs, targets = inputs.to(device), targets.to(device)
            except StopIteration:
                # 如果迭代器用完，重新创建
                self.train_loader, _ = self._load_dataset('./data', self.batch_size)
                inputs, targets = next(iter(self.train_loader))
                inputs, targets = inputs.to(device), targets.to(device)
            
            # 4. 计算梯度
            gradients, loss = self.compute_gradients(inputs, targets)
            
            # 5. 将梯度发送到参数服务器
            start_time = time.time()
            new_version = invoke(self.ps_instance, keep_alive, 
                                "update_params", gradients, learning_rate=0.01)
            update_time = (time.time() - start_time) * 1000
            
            # 6. 更新训练状态
            self.global_step += 1
            self.train_losses.append(loss)
            
            return {
                "player_id": self.player_id,
                "model_type": self.model_type,
                "step": self.global_step,
                "loss": loss,
                "param_version": new_version,
                "pull_time_ms": pull_time,
                "update_time_ms": update_time
            }
            
        except Exception as e:
            print(f"玩家 {self.player_id} 训练批次失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                "player_id": self.player_id,
                "error": str(e)
            }
    
    def validate(self) -> Dict[str, Any]:
        """在验证集上评估模型"""
        self.model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100.0 * correct / total
        avg_loss = val_loss / len(self.val_loader)
        
        self.val_accuracies.append(accuracy)
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
        
        self.model.train()
        return {
            "player_id": self.player_id,
            "model_type": self.model_type,
            "accuracy": accuracy,
            "val_loss": avg_loss,
            "best_accuracy": self.best_accuracy
        }
    
    def save_checkpoint(self, checkpoint_dir: str) -> str:
        """保存模型检查点"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            checkpoint_dir, 
            f"{self.model_type}_{self.player_id}_step{self.global_step}.pth"
        )
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'best_accuracy': self.best_accuracy,
            'train_losses': self.train_losses,
            'val_accuracies': self.val_accuracies
        }
        
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """加载模型检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_accuracy = checkpoint['best_accuracy']
        self.train_losses = checkpoint['train_losses']
        self.val_accuracies = checkpoint['val_accuracies']
        
        print(f"玩家 {self.player_id} 从检查点 {checkpoint_path} 恢复，步骤: {self.global_step}")


class PyTorchDistributedTrainer:
    """PyTorch分布式训练系统"""
    
    def __init__(self, model_types: List[str], num_players_per_model: int = 2, 
                 batch_size: int = 32, dataset_path: str = './data'):
        """
        初始化分布式训练系统
        
        参数:
            model_types: 要训练的模型类型列表
            num_players_per_model: 每种模型的玩家数量
            batch_size: 批处理大小
            dataset_path: 数据集路径
        """
        self.model_types = model_types
        self.num_players_per_model = num_players_per_model
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        
        # 存储模型服务器和玩家
        self.model_servers = {}
        self.players = []
        
        print(f"初始化分布式训练系统，模型类型: {model_types}，每种模型 {num_players_per_model} 个玩家")
    
    def _calculate_model_size_mb(self, model: nn.Module) -> float:
        """计算模型参数大小（MB）"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size_bytes = param_size + buffer_size
        total_size_mb = total_size_bytes / (1024 * 1024)
        
        return total_size_mb
    
    def initialize(self):
        """初始化参数服务器和模型玩家"""
        print("初始化参数服务器和模型玩家...")
        
        # 为每种模型创建一个参数服务器
        for model_type in self.model_types:
            print(f"为模型 {model_type} 创建参数服务器...")
            
            # 创建临时模型以获取参数形状和大小
            temp_model = None
            try:
                # 创建一个临时玩家来获取模型
                temp_player = ModelPlayerActor(
                    player_id="temp",
                    model_type=model_type,
                    model_id="temp",
                    ps_instance=None,
                    batch_size=self.batch_size
                )
                temp_model = temp_player.model
                
                # 获取参数字典
                param_dict = temp_player.model_to_param_dict()
                
                # 计算参数形状
                param_shapes = {name: param.shape for name, param in param_dict.items()}
                
                # 计算模型大小（MB）
                model_size_mb = self._calculate_model_size_mb(temp_model)
                print(f"模型 {model_type} 大小: {model_size_mb:.2f}MB")
                
                # 使用唯一ID避免命名冲突
                model_id = f"{model_type}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
                
                # 初始化参数服务器，只创建一次绑定实例
                ps_instance = PSActor.bind(model_id, model_id, param_shapes)
                ps_instance.options(
                    unique_id=model_id, 
                    is_persistent=True,
                    state_memory_size=int(model_size_mb * 1.2)  # 增加20%的缓冲
                )
                
                # 存储模型服务器信息
                self.model_servers[model_type] = {
                    'model_id': model_id,
                    'ps_instance': ps_instance,
                    'param_shapes': param_shapes,
                    'model_size_mb': model_size_mb
                }
                
                print(f"模型 {model_type} 参数服务器初始化完成，ID: {model_id}")
                
            except Exception as e:
                print(f"初始化模型 {model_type} 参数服务器失败: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            finally:
                # 清理临时模型
                if temp_model is not None:
                    del temp_model
                    torch.cuda.empty_cache()
        
        # 为每种模型创建玩家
        for model_type in self.model_types:
            if model_type not in self.model_servers:
                print(f"跳过模型 {model_type}，参数服务器初始化失败")
                continue
                
            server_info = self.model_servers[model_type]
            
            for i in range(self.num_players_per_model):
                player_id = f"player_{model_type}_{i+1}"
                
                try:
                    # 创建模型玩家
                    player = ModelPlayerActor(
                        player_id=player_id,
                        model_type=model_type,
                        model_id=server_info['model_id'],
                        ps_instance=server_info['ps_instance'],
                        dataset_path=self.dataset_path,
                        batch_size=self.batch_size
                    )
                    
                    self.players.append(player)
                    print(f"玩家 {player_id} 初始化完成")
                    
                except Exception as e:
                    print(f"初始化玩家 {player_id} 失败: {e}")
                    continue
        
        print(f"初始化完成，共 {len(self.players)} 个玩家")
    
    def train(self, num_iterations: int = 1000, validate_every: int = 100, 
              checkpoint_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        训练所有模型
        
        参数:
            num_iterations: 训练迭代次数
            validate_every: 每多少次迭代验证一次
            checkpoint_dir: 检查点保存目录
        """
        if not self.players:
            self.initialize()
        
        if not self.players:
            print("没有可用的玩家，训练终止")
            return {"error": "没有可用的玩家"}
        
        print(f"开始训练 {len(self.players)} 个模型玩家，共 {num_iterations} 次迭代")
        
        # 训练统计
        training_stats = {player.player_id: [] for player in self.players}
        validation_stats = {player.player_id: [] for player in self.players}
        
        # 记录开始时间
        start_time = time.time()
        
        # 主训练循环
        for iteration in range(num_iterations):
            iteration_start = time.time()
            
            # 使用线程池并行训练所有玩家
            with ThreadPoolExecutor(max_workers=len(self.players)) as executor:
                # 最后一个玩家不保持参数服务器存活
                futures = []
                
                for i, player in enumerate(self.players):
                    # 同一模型的最后一个玩家不保持参数服务器存活
                    is_last_player = i == len(self.players) - 1 or (
                        i + 1 < len(self.players) and 
                        player.model_type != self.players[i + 1].model_type
                    )
                    
                    # keep_alive = not is_last_player
                    keep_alive = True
                    futures.append(executor.submit(player.train_batch, keep_alive))
                
                # 收集结果
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if 'player_id' in result and 'error' not in result:
                            training_stats[result['player_id']].append(result)
                    except Exception as e:
                        print(f"训练批次失败: {e}")
            
            # 计算当前迭代的统计信息
            iteration_time = time.time() - iteration_start
            current_losses = []
            
            for player in self.players:
                if player.train_losses:
                    current_losses.append(player.train_losses[-1])
            
            avg_loss = np.mean(current_losses) if current_losses else float('inf')
            
            # 打印进度
            if (iteration + 1) % 10 == 0 or iteration == 0 or iteration == num_iterations - 1:
                print(f"迭代 {iteration+1}/{num_iterations}, 平均损失: {avg_loss:.4f}, "
                      f"耗时: {iteration_time:.2f}秒")
            
            # 定期验证
            if (iteration + 1) % validate_every == 0 or iteration == num_iterations - 1:
                print(f"\n===== 迭代 {iteration+1} 验证 =====")
                
                for player in self.players:
                    val_result = player.validate()
                    validation_stats[player.player_id].append(val_result)
                    
                    print(f"玩家 {player.player_id} ({player.model_type}): "
                          f"准确率: {val_result['accuracy']:.2f}%, "
                          f"最佳: {val_result['best_accuracy']:.2f}%")
                
                # 保存检查点
                if checkpoint_dir:
                    for player in self.players:
                        checkpoint_path = player.save_checkpoint(checkpoint_dir)
                        print(f"玩家 {player.player_id} 检查点已保存: {checkpoint_path}")
        
        # 训练结束
        total_time = time.time() - start_time
        
        # 收集最终统计信息
        final_stats = {
            'total_time': total_time,
            'iterations': num_iterations,
            'iterations_per_second': num_iterations / total_time,
            'models': {}
        }
        
        for model_type in self.model_types:
            if model_type not in self.model_servers:
                continue
                
            model_players = [p for p in self.players if p.model_type == model_type]
            if not model_players:
                continue
                
            model_stats = {
                'players': [p.player_id for p in model_players],
                'best_accuracy': max(p.best_accuracy for p in model_players),
                'final_loss': np.mean([p.train_losses[-1] if p.train_losses else float('inf') 
                                      for p in model_players]),
                'model_size_mb': self.model_servers[model_type]['model_size_mb']
            }
            final_stats['models'][model_type] = model_stats
        
        print("\n========== 训练完成 ==========")
        print(f"总训练时间: {total_time:.2f}秒")
        print(f"平均每秒迭代: {num_iterations / total_time:.2f}")
        
        for model_type, stats in final_stats['models'].items():
            print(f"\n模型 {model_type}:")
            print(f"  模型大小: {stats['model_size_mb']:.2f}MB")
            print(f"  最佳准确率: {stats['best_accuracy']:.2f}%")
            print(f"  最终损失: {stats['final_loss']:.4f}")
        
        # 保存训练统计信息
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            stats_path = os.path.join(checkpoint_dir, f"training_stats_{int(time.time())}.json")
            
            with open(stats_path, 'w') as f:
                json.dump({
                    'final_stats': final_stats,
                    'training_stats': {k: v for k, v in training_stats.items() if v},
                    'validation_stats': {k: v for k, v in validation_stats.items() if v}
                }, f, indent=2)
            
            print(f"\n训练统计信息已保存: {stats_path}")
        
        return final_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch分布式训练系统")
    parser.add_argument("--models", type=str, default="resnet18,alexnet", 
                        help="要训练的模型类型，用逗号分隔")
    parser.add_argument("--players", type=int, default=2, 
                        help="每种模型的玩家数量")
    parser.add_argument("--batch-size", type=int, default=32, 
                        help="批处理大小")
    parser.add_argument("--iterations", type=int, default=500, 
                        help="训练迭代次数")
    parser.add_argument("--validate-every", type=int, default=100, 
                        help="每多少次迭代验证一次")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", 
                        help="检查点保存目录")
    
    args = parser.parse_args()
    
    # 解析模型类型
    model_types = args.models.split(',')
    
    # 初始化Ray（如果尚未初始化）
    if not ray.is_initialized():
        ray.init()
    
    # 创建训练器并开始训练
    trainer = PyTorchDistributedTrainer(
        model_types=model_types,
        num_players_per_model=args.players,
        batch_size=args.batch_size
    )
    
    # 执行训练
    trainer.train(
        num_iterations=args.iterations,
        validate_every=args.validate_every,
        checkpoint_dir=args.checkpoint_dir
    ) 