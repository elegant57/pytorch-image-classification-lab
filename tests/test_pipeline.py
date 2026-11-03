import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset
from config import Config
from data_loader import get_dataloaders
from models import CustomNet, CIFAR100Net
from trainer import Trainer


class PipelineTests(unittest.TestCase):
    def test_model_shapes_and_config(self):
        with torch.no_grad():
            self.assertEqual(CustomNet(2)(torch.zeros(2, 3, 32, 32)).shape, (2, 2))
            self.assertEqual(CIFAR100Net().eval()(torch.zeros(1, 3, 32, 32)).shape, (1, 100))
        self.assertEqual(Config(['--dataset', 'CIFAR100']).num_classes, 100)

    def test_custom_data_resize_and_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            for split in ('train', 'val'):
                for cls in ('ants', 'bees'):
                    folder = Path(tmp) / split / cls
                    folder.mkdir(parents=True)
                    Image.new('RGB', (50, 40)).save(folder / 'one.jpg')
            config = Config(['--dataset', 'Custom', '--data_dir', tmp, '--num_classes', '2', '--num_workers', '0'])
            train, val, n, m = get_dataloaders(config)
            self.assertEqual((n, m), (2, 2))
            self.assertEqual(next(iter(val))[0].shape, (2, 3, 32, 32))
            (Path(tmp) / 'val' / 'ants').rename(Path(tmp) / 'val' / 'other')
            with self.assertRaisesRegex(ValueError, 'same class'):
                get_dataloaders(config)

    def test_training_checkpoint_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            torch.manual_seed(42)
            data = TensorDataset(torch.randn(5, 3, 4, 4), torch.tensor([0, 1, 0, 1, 0]))
            loader = DataLoader(data, batch_size=2)
            cfg = SimpleNamespace(get_device=lambda: torch.device('cpu'), optimizer='SGD',
                learning_rate=0.01, momentum=0.9, weight_decay=0.0,
                log_dir=tmp + '/logs', save_dir=tmp, epochs=1, save_interval=1, model='tiny')
            factory = lambda: torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48, 2))
            trainer = Trainer(cfg, factory(), loader, loader, 5, 5)
            trainer.train()
            checkpoint = Path(tmp) / 'tiny_epoch_1.pth'
            saved = torch.load(checkpoint, weights_only=True)
            self.assertEqual(saved['best_accuracy'], trainer.best_accuracy)
            cfg.epochs = 2
            resumed = Trainer(cfg, factory(), loader, loader, 5, 5)
            self.assertEqual(resumed.resume(checkpoint), 1)
            resumed.train()
            self.assertTrue((Path(tmp) / 'tiny_epoch_2.pth').is_file())
            self.assertEqual(resumed.start_epoch, 1)


if __name__ == '__main__':
    unittest.main()
