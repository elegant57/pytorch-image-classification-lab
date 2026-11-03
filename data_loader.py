"""CIFAR and class-folder datasets with a shared 32 x 32 input contract."""
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class CustomDataset(datasets.ImageFolder):
    def __init__(self, root_dir, transform=None):
        super().__init__(root_dir, transform=transform)


def get_transforms():
    normalize = transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    train = transforms.Compose([
        transforms.RandomResizedCrop(32), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize,
    ])
    test = transforms.Compose([
        transforms.Resize((32, 32)), transforms.ToTensor(), normalize,
    ])
    return train, test


def get_dataloaders(config):
    train_transform, test_transform = get_transforms()
    if config.dataset in ('CIFAR10', 'CIFAR100'):
        cls = getattr(datasets, config.dataset)
        train = cls(root=config.data_dir, train=True, download=True, transform=train_transform)
        test = cls(root=config.data_dir, train=False, download=True, transform=test_transform)
    elif config.dataset == 'Custom':
        train = CustomDataset(Path(config.data_dir) / 'train', train_transform)
        test = CustomDataset(Path(config.data_dir) / 'val', test_transform)
        if train.class_to_idx != test.class_to_idx:
            raise ValueError('train/val must contain the same class directories')
        if len(train.classes) != config.num_classes:
            raise ValueError('--num_classes must match the number of class directories')
    else:
        raise ValueError(f'Unsupported dataset: {config.dataset}')
    options = dict(batch_size=config.batch_size, num_workers=config.num_workers)
    return (DataLoader(train, shuffle=True, **options),
            DataLoader(test, shuffle=False, **options), len(train), len(test))
