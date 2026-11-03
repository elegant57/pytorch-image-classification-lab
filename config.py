import torch
import argparse

class Config:
    def __init__(self, argv=None):
        self.parser = argparse.ArgumentParser(description='Neural Network Training Framework')
        self._add_arguments()
        self.args = self.parser.parse_args(argv)
        if self.args.num_classes is None:
            self.args.num_classes = 100 if self.args.dataset == "CIFAR100" else 10
        expected = {"CIFAR10": 10, "CIFAR100": 100}.get(self.args.dataset)
        if expected and self.args.num_classes != expected:
            self.parser.error(f"{self.args.dataset} requires --num_classes {expected}")
        if min(self.args.batch_size, self.args.epochs, self.args.save_interval) < 1:
            self.parser.error("batch_size, epochs and save_interval must be positive")
        if self.args.num_workers < 0:
            self.parser.error("num_workers must be nonnegative")
    
    def _add_arguments(self):
        # 数据集配置
        self.parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'Custom'], help='Dataset to use')
        self.parser.add_argument('--data_dir', type=str, default='data/data_CIFAR', help='Path to dataset')
        self.parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
        self.parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
        
        # 模型配置
        self.parser.add_argument('--model', type=str, default='custom', choices=['vgg16', 'resnet18', 'custom'], help='Model architecture')
        self.parser.add_argument('--num_classes', type=int, default=None, help='Number of classes')
        
        # 训练配置
        self.parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
        self.parser.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate')
        self.parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'], help='Optimizer')
        self.parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
        self.parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
        
        # 设备配置
        self.parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='Device to use')
        
        # 保存和日志配置
        self.parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
        self.parser.add_argument('--log_dir', type=str, default='runs', help='Directory for TensorBoard logs')
        self.parser.add_argument('--save_interval', type=int, default=1, help='Save model every N epochs')
        
        # 其他配置
        self.parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training')
        self.parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    def get_device(self):
        if self.args.device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.args.device)
    
    def __getattr__(self, name):
        return getattr(self.args, name)

