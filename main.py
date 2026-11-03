import torch
import random
import numpy as np
from config import Config
from data_loader import get_dataloaders
from models import get_model
from trainer import Trainer

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    config = Config()
    # 设置随机种子
    set_seed(config.seed)
    
    # 打印配置信息
    print("===== Training Configuration =====")
    print(f"Dataset: {config.dataset}")
    print(f"Model: {config.model}")
    print(f"Number of classes: {config.num_classes}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Optimizer: {config.optimizer}")
    print(f"Device: {config.get_device()}")
    print("=================================")
    
    # 加载数据
    print("\nLoading data...")
    train_dataloader, test_dataloader, train_size, test_size = get_dataloaders(config)
    print(f"Train dataset size: {train_size}")
    print(f"Test dataset size: {test_size}")
    
    # 创建模型
    print("\nCreating model...")
    model = get_model(config)
    print(f"Model: {config.model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 初始化训练器
    trainer = Trainer(
        config=config,
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        train_size=train_size,
        test_size=test_size
    )
    
    # 恢复训练（如果需要）
    if config.resume:
        trainer.resume(config.resume)
    
    # 开始训练
    print("\nStarting training...")
    trainer.train()