import argparse
import base64
import os
from io import BytesIO

import mlflow
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from pytorch_lightning.metrics.functional import accuracy, f1 as f1_score, fbeta
from torch import nn
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import models

from liveness_detection.config import CONDA_ENV

MODEL_STATE = 'model_state'


def decode_and_resize_image(raw_bytes, size):
    """
    Read, decode and resize raw image bytes (e.g. raw content of a jpeg file).
    :param raw_bytes: Image bits, e.g. jpeg image.
    :param size: requested output dimensions
    :return: Multidimensional numpy array representing the resized image.
    """
    return np.asarray(Image.open(BytesIO(raw_bytes)).resize(size), dtype=np.float32)


class LivenessDetector(pl.LightningModule):
    def __init__(self, **kwargs):
        """
        Initializes the network
        """
        super().__init__()
        self.series_len = kwargs.get('channels', 5)

        from efficientnet_pytorch import EfficientNet
        self.backbone = EfficientNet.from_pretrained(kwargs.get('network', 'efficientnet-b0'),
                                                     num_classes=1,
                                                     in_channels=self.series_len)
        # self.backbone = models.resnet18(True)
        # self.backbone.fc = nn.Linear(512, 1)
        # self.backbone.conv1 = nn.Conv2d(self.series_len, 64, 7, 2, bias=False)

        self.swa_model = torch.optim.swa_utils.AveragedModel(self.backbone)

        self.model = self.backbone

        self.train_live_file = kwargs.get('tl')
        self.test_live_file = kwargs.get('vl')
        self.train_spoofed_file = kwargs.get('ts')
        self.test_spoofed_file = kwargs.get('vs')

        self.lr = kwargs.get('lr', 0.02)
        self.epochs = kwargs.get('max_epochs', 100)
        self.swa_start = int(self.epochs * 0.75)
        self.cycle_period = 35

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.optimizer = None
        self.scheduler = None
        self.metrics = {}

    def forward(self, x):
        x = self.model(x)

        return x

    def training_step(self, batch, idx, swa=False):
        x, y = batch
        if swa:
            logits = self.swa_model(x)
        else:
            logits = self.backbone(x)
        loss = self.bce_loss(logits, y)
        scores = torch.sigmoid(logits)

        scores = scores.gt(0.5).float()
        f1 = fbeta(scores, y, 1, average='macro', beta=0.7)
        acc = accuracy(scores, y, 2, class_reduction='macro')

        return {"loss": loss, 'f1': f1, 'acc': acc}

    def validation_step(self, batch, idx):
        res = self.training_step(batch, idx)

        return res

    def training_epoch_end(self, outputs):
        mean_loss = torch.stack([o['loss'] for o in outputs]).mean()
        mean_f1 = torch.stack([o['f1'] for o in outputs]).mean()
        mean_acc = torch.stack([o['acc'] for o in outputs]).mean()
        self.log('loss', mean_loss)
        self.log('f1', mean_f1)
        self.log('acc', mean_acc)

        self.log('lr', self.optimizer.param_groups[0]['lr'])

        # if (self.current_epoch + 1) % self.cycle_period == 0:
        #     self.swa_model.update_parameters(self.backbone)
        # self.scheduler.step

        if self.current_epoch > self.swa_start:
            self.swa_model.update_parameters(self.backbone)
            self.swa_scheduler.step()
        else:
            self.scheduler.step()

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([o['loss'] for o in outputs]).mean()
        mean_f1 = torch.stack([o['f1'] for o in outputs]).mean()
        mean_acc = torch.stack([o['acc'] for o in outputs]).mean()
        self.metrics['val_f1'] = mean_f1
        self.metrics['val_loss'] = mean_loss
        self.metrics['val_acc'] = mean_acc

        self.metrics.setdefault('best_loss', mean_loss)
        self.metrics.setdefault('best_f1', mean_f1)
        self.metrics.setdefault('best_acc', mean_acc)

        if self.metrics['best_loss'] > mean_loss:
            self.metrics['best_loss'] = mean_loss

        if self.metrics['best_f1'] < mean_f1:
            self.metrics['best_f1'] = mean_f1

        if self.metrics['best_acc'] < mean_acc:
            self.metrics['best_acc'] = mean_acc

        self.log_dict(self.metrics)

    def test_step(self, batch, idx):
        res = self.training_step(batch, idx, True)

        return res

    def test_epoch_end(self, outputs):
        mean_loss = torch.stack([o['loss'] for o in outputs]).mean()
        mean_f1 = torch.stack([o['f1'] for o in outputs]).mean()
        mean_acc = torch.stack([o['acc'] for o in outputs]).mean()
        self.log('swa_final_loss', mean_loss)
        self.log('swa_final_f1', mean_f1)
        self.log('swa_final_acc', mean_acc)

    def on_train_end(self):
        torch.optim.swa_utils.update_bn(self.train_dataloader(), self.swa_model, self.device)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.epochs)
        self.swa_scheduler = torch.optim.swa_utils.SWALR(self.optimizer, 0.03, anneal_epochs=int(self.epochs * 0.25))

        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, self.cycle_period)

        return [self.optimizer]

    def setup(self, stage: str):
        from liveness_detection.dataset import LivenessDataset
        from liveness_detection.augmentation import Preprocessor
        preprocessor = Preprocessor(augment=True)
        self._test_dataset = LivenessDataset(self.test_live_file,
                                             self.test_spoofed_file,
                                             preprocessor, self.series_len)
        self._train_dataset = LivenessDataset(self.train_live_file,
                                              self.train_spoofed_file,
                                              preprocessor, self.series_len)

    def train_dataloader(self):
        return DataLoader(self._train_dataset, batch_size=32, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self._test_dataset, batch_size=32, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self._test_dataset, batch_size=32, num_workers=8)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--network', default='efficientnet-b0', help='efficientnet-b[0-6]')
        parser.add_argument('--image_width', type=int, default=128)
        parser.add_argument('--channels', type=int, default=5)

        parser.add_argument('--tl', help='train live file path', default='../data/train_live.txt')
        parser.add_argument('--vl', help='val live file path', default='../data/test_live.txt')
        parser.add_argument('--ts', help='train spoofed file path', default='../data/train_spoofed.txt')
        parser.add_argument('--vs', help='val spoofed file path', default='../data/test_spoofed.txt')
        return parser


class LivenessDetectorWrapper(mlflow.pyfunc.PythonModel):

    def __init__(self, map_location=None, input_dim=(5, 128, 128, 3), use_swa=True):
        super(LivenessDetectorWrapper, self).__init__()
        self.map_location = map_location
        self.input_dim = input_dim
        self.use_swa = use_swa

    def load_context(self, context):
        self._detector = LivenessDetector.load_from_checkpoint(context.artifacts[MODEL_STATE],
                                                               map_location=self.map_location)
        self._detector.eval()
        self._detector.freeze()

        if self.use_swa:
            self._detector.model = self._detector.swa_model
        else:
            self._detector.model = self._detector.backbone

        from liveness_detection.augmentation import Preprocessor
        self._preprocessor = Preprocessor()

    def predict(self, context, model_input):
        def decode_img(x):
            r = base64.decodebytes(bytes(x, encoding='utf-8'))
            q = np.frombuffer(r, dtype='uint8').reshape(self.input_dim)
            return q

        images = [self._preprocessor(decode_img(x)).unsqueeze(0) for x in model_input['image'].values]

        input = torch.cat(images, dim=0)
        with torch.no_grad():
            output = torch.sigmoid(self._detector(input))
            return output.cpu().numpy().reshape((-1))

    @staticmethod
    def export_model(model_path, **kwargs):
        mlflow.pyfunc.log_model(kwargs.get('name', 'model'),
                                python_model=LivenessDetectorWrapper(kwargs.get('map_location', 'cpu'),
                                                                     (kwargs.get('channels', 5),
                                                                      kwargs.get('image_width', 128),
                                                                      kwargs.get('image_width', 128),
                                                                      3),
                                                                     use_swa=kwargs.get('use_swa', True)),
                                conda_env=CONDA_ENV,
                                artifacts={MODEL_STATE: model_path},
                                code_path=['./',
                                           '../utils'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--progress_bar_refresh_rate', type=int, default=20)
    parser.add_argument('--default_root_dir', default=os.path.dirname(os.getcwd()), help='pytorch-lightning log path')

    parser.add_argument('--map_location', default='cuda')

    parser = LivenessDetector.add_model_specific_args(parser)

    args = parser.parse_args()
    mlflow.set_tracking_uri('file:../mlruns')
    mlflow.set_experiment('liveness')
    mlflow.pytorch.autolog(log_every_n_epoch=1)
    with mlflow.start_run():
        mlflow.log_param('image_width', args.image_width)
        mlflow.log_param('channels', args.channels)
        mlflow.log_param('netowork', args.network)
        mlflow.log_param('channels', args.channels)

        model = LivenessDetector(**vars(args))
        trainer = pl.Trainer.from_argparse_args(args)
        trainer.fit(model)
        trainer.test(model)

        swa_path = '../models/swa.ckpt'
        trainer.save_checkpoint(swa_path)
        LivenessDetectorWrapper.export_model(swa_path, **vars(args), name='swa', use_swa=True)
        LivenessDetectorWrapper.export_model(swa_path, **vars(args), name='model', use_swa=False)
