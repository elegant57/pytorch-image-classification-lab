import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import time

class Trainer:
    def __init__(self, config, model, train_dataloader, test_dataloader, train_size, test_size):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.train_size = train_size
        self.test_size = test_size
        self.device = config.get_device()
        
        # 移动模型到设备
        self.model.to(self.device)
        
        # 损失函数
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn.to(self.device)
        
        # 优化器
        if config.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(
                params=model.parameters(),
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay
            )
        elif config.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(
                params=model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        
        # 学习率调度器（可选）
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=0.1
        )
        
        # 日志记录
        self.writer = SummaryWriter(log_dir=config.log_dir)
        
        # 创建保存目录
        os.makedirs(config.save_dir, exist_ok=True)
        
        # 训练状态
        self.best_accuracy = 0.0
        self.start_epoch = 0
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(self.train_dataloader):
            # 移动数据到设备
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # 前向传播
            outputs = self.model(images)
            loss = self.loss_fn(outputs, targets)
            
            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            
            # 打印训练信息
            if (batch_idx + 1) % 100 == 0:
                print(f'[Epoch {epoch+1}, Batch {batch_idx+1}/{len(self.train_dataloader)}] Loss: {loss.item():.4f}')
        
        # 计算平均损失
        avg_loss = total_loss / self.train_size
        return avg_loss
    
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        
        with torch.no_grad():
            for images, targets in self.test_dataloader:
                # 移动数据到设备
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)
                
                total_loss += loss.item() * images.size(0)
                
                # 计算准确率
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
        
        # 计算平均损失和准确率
        avg_loss = total_loss / self.test_size
        accuracy = 100.0 * correct / self.test_size
        
        return avg_loss, accuracy
    
    def train(self):
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.config.epochs):
            print(f"\n------------ Epoch {epoch+1}/{self.config.epochs} ------------")
            
            # 训练一个epoch
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_accuracy = self.validate(epoch)
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录日志
            self.writer.add_scalar('Loss/Train', train_loss, epoch+1)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch+1)
            self.writer.add_scalar('Accuracy/Validation', val_accuracy, epoch+1)
            self.writer.add_scalar('Learning Rate', self.optimizer.param_groups[0]['lr'], epoch+1)
            
            # 打印 epoch 结果
            print(f"Epoch {epoch+1} Results:")
            print(f"- Train Loss: {train_loss:.4f}")
            print(f"- Validation Loss: {val_loss:.4f}")
            print(f"- Validation Accuracy: {val_accuracy:.2f}%")
            print(f"- Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            is_best = val_accuracy > self.best_accuracy
            self.best_accuracy = max(self.best_accuracy, val_accuracy)

            # 保存模型
            if (epoch + 1) % self.config.save_interval == 0:
                checkpoint_path = os.path.join(self.config.save_dir, f'{self.config.model}_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_accuracy': self.best_accuracy,
                }, checkpoint_path)
                print(f"Model saved at: {checkpoint_path}")
            
            # 保存最佳模型
            if is_best:
                self.best_accuracy = val_accuracy
                best_model_path = os.path.join(self.config.save_dir, f'{self.config.model}_best.pth')
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_accuracy': self.best_accuracy,
                }, best_model_path)
                print(f"Best model saved at: {best_model_path}")
        
        # 训练结束
        end_time = time.time()
        print(f"\nTraining completed in {end_time - start_time:.2f} seconds")
        print(f"Best validation accuracy: {self.best_accuracy:.2f}%")
        
        # 关闭日志记录
        self.writer.close()
    
    def resume(self, checkpoint_path):
        """从检查点恢复训练"""
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_accuracy = checkpoint['best_accuracy']
        self.start_epoch = checkpoint['epoch']
        print(f'Resumed from epoch {self.start_epoch}: {checkpoint_path}')
        return self.start_epoch
