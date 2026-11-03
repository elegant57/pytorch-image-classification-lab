# 整理记录

原始 Python 文件在本次 Codex 工作目录的 work/original_sources 中备份。数据、权重未删除，未执行 GitHub 发布。

## 路径迁移

- `Complete_train.py` → `examples/Complete_train.py`
- `dataset_transforms.py` → `examples/dataset_transforms.py`
- `first_demo.py` → `examples/first_demo.py`
- `Linear_test.py` → `examples/Linear_test.py`
- `model_load.py` → `examples/model_load.py`
- `model_pretrained.py` → `examples/model_pretrained.py`
- `model_save.py` → `examples/model_save.py`
- `nn_Loss_optim.py` → `examples/nn_Loss_optim.py`
- `own_test.py` → `examples/own_test.py`
- `ply.py` → `examples/ply.py`
- `practice_before_convlution.py` → `examples/practice_before_convlution.py`
- `practice_before_convlution2.py` → `examples/practice_before_convlution2.py`
- `practice_Complete.py` → `examples/practice_Complete.py`
- `practice_MaxPool_NoLinear.py` → `examples/practice_MaxPool_NoLinear.py`
- `practice_MaxPool_Nolinear2.py` → `examples/practice_MaxPool_Nolinear2.py`
- `practice_sequential.py` → `examples/practice_sequential.py`
- `Review_Complement.py` → `examples/Review_Complement.py`
- `Review_Test.py` → `examples/Review_Test.py`
- `test_Tensorboard.py` → `examples/test_Tensorboard.py`
- `torch_nn(Convolution).py` → `examples/torch_nn(Convolution).py`
- `torch_nn(MaxPool).py` → `examples/torch_nn(MaxPool).py`
- `torch_nn(sequential).py` → `examples/torch_nn(sequential).py`
- `trained_test.py` → `examples/trained_test.py`
- `Transforms_test.py` → `examples/Transforms_test.py`
- `test/call_test.py` → `examples/call_test.py`
- `CIFAR10_logs` → `runs/legacy/CIFAR10_logs`
- `CIFAR100_logs` → `runs/legacy/CIFAR100_logs`
- `Complete_train` → `runs/legacy/Complete_train`
- `logs_tran` → `runs/legacy/logs_tran`
- `Neural_Network` → `runs/legacy/Neural_Network`
- `practice` → `runs/legacy/practice`
- `practice_Complete` → `runs/legacy/practice_Complete`

## 运行行为调整

- CLI 配置仅在入口解析；默认 custom 模型，CIFAR 类别数自动匹配。
- 自定义分类数据用 ImageFolder 读取，校验类别映射，并将评估输入统一到 32×32。
- 损失按样本加权；恢复轮次、跨设备加载及检查点中的最佳准确率已修正。
- 历史练习逻辑保持原样。
