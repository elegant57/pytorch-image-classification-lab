# PyTorch Image Classification Lab

从 PyTorch 基础练习逐步整理出的图像分类实验项目。主流程将数据加载、模型构建、训练评估和检查点管理拆分为独立模块，支持 CIFAR-10、CIFAR-100 与按类别文件夹组织的自定义数据集。

本项目适合展示对分类训练流程和 PyTorch 工程基础的理解；历史练习保存在 `examples/`，可结合主流程对照阅读。

## 已实现功能

- 数据：CIFAR 自动下载；自定义图片按 `train/类别`、`val/类别` 读取，检查类别映射一致性。
- 模型：自定义 CNN、VGG16、ResNet18；自定义 CNN 在 100 类配置下使用 `CIFAR100Net`。
- 训练：交叉熵损失、SGD/Adam、StepLR 调度器、CPU/CUDA 选择、随机种子设置。
- 评估：每轮统计样本加权平均损失和 Top-1 Accuracy，使用 `eval()` 与 `no_grad()`。
- 记录：TensorBoard 损失、准确率与学习率曲线。
- 保存：周期检查点、最佳模型；恢复模型、优化器、调度器、最佳准确率和已完成轮次。

VGG16 和 ResNet18 均使用 `weights=None`，属于从头训练，不是预训练迁移学习。所有主流程模型统一接收 32×32 RGB 输入；VGG16/ResNet18 保留 torchvision 原始结构，尚未针对 CIFAR 修改 stem。

## 目录

```text
.
├── main.py              # 训练入口、种子与运行配置
├── config.py            # 命令行参数及基础校验
├── data_loader.py       # CIFAR / ImageFolder 与变换
├── models.py            # 自定义 CNN、VGG16、ResNet18
├── trainer.py           # 训练、评估、日志和检查点
├── examples/            # 原始基础学习与实验脚本
├── tests/               # 离线回归检查
├── docs/                # 面试讲解与整理说明
├── data/README.md       # 本地数据说明；实际数据不上传
├── requirements.txt
└── .gitignore
```

本地已有 `.pth` 权重仍保留原路径，供历史脚本使用；`runs/legacy/` 保存旧日志。二者均被 Git 忽略，上传包也不包含这些产物。

## 环境与快速开始

本地验证环境：Python 3.13.5、torch 2.8.0+cu128、torchvision 0.23.0+cu128。依赖清单未锁定具体版本；安装时应使用相互匹配的 torch/torchvision 版本。以下命令均在仓库根目录运行。

```bash
python -m venv .venv
# Windows PowerShell 激活：.venv\Scripts\Activate.ps1
# Linux/macOS 激活：source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py --help
```

首次使用 CIFAR 需要下载数据。先用小型自定义 CNN 运行：

```bash
python main.py --dataset CIFAR10 --model custom --epochs 10 --num_workers 0
python main.py --dataset CIFAR100 --model custom --epochs 10 --num_workers 0
python main.py --dataset CIFAR10 --model resnet18 --optimizer Adam --learning_rate 0.001 --num_workers 0
```

默认模型为 `custom`；CIFAR-10/100 分别自动选择 10/100 类。Windows 初次运行建议使用 `--num_workers 0`，之后可按机器配置增加。

### 自定义分类数据

```text
data/my_dataset/
├── train/
│   ├── ants/*.jpg
│   └── bees/*.jpg
└── val/
    ├── ants/*.jpg
    └── bees/*.jpg
```

```bash
python main.py --dataset Custom --data_dir data/my_dataset --num_classes 2 --model custom --num_workers 0
```

训练集使用随机裁剪、水平翻转与归一化；评估集缩放为 32×32 并使用相同归一化参数。类别顺序按目录名排序，两个划分必须包含相同类别目录。

### 日志与恢复训练

```bash
tensorboard --logdir runs
python main.py --dataset CIFAR10 --model custom --epochs 20 --num_workers 0 --resume checkpoints/custom_epoch_10.pth
```

`--epochs 20` 表示总共完成 20 轮；从第 10 轮恢复时继续第 11–20 轮。恢复时需要使用与检查点一致的模型、类别数和优化器。恢复功能不保存随机数生成器状态，因此不保证逐步复现一次不中断运行的结果。检查点仅包含训练状态，不包含数据集或完整 CLI 配置，比较实验时应同时保存运行命令。

常用参数：`--batch_size`（64）、`--learning_rate`（0.01）、`--optimizer`（SGD）、`--epochs`（10）、`--device`（auto）、`--save_dir`（checkpoints）、`--log_dir`（runs）、`--save_interval`（1）、`--seed`（42）。

## 验证与结果边界

```bash
python -m unittest discover -s tests -v
```

检查覆盖模型输出维度、自定义数据集输入和类别映射，以及合成数据上的训练、检查点与恢复轮次。测试无需下载数据。

仓库暂不提供可追溯到固定配置的 CIFAR 准确率表，不将本地历史权重或日志直接作为性能结论。主流程每轮使用 CIFAR 官方测试集评估，当前没有独立验证集；如需正式比较模型，应从训练集划出验证集，并仅在最终报告时使用测试集。尚未实现独立推理 CLI、早停、多卡训练或部署服务。

## 来源与许可

这是个人学习与实验整理项目。原始目录未附教程来源或开源许可证，仓库不额外宣称代码全部原创；如有课程或教程引用，应在发布时补充真实来源。CIFAR、torchvision 模型及自备图片的许可由各自来源决定，数据与历史权重不随仓库发布。
