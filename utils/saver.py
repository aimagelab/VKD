import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
import json

from torch.utils.tensorboard import SummaryWriter
from model.net import TriNet


class Saver(object):
    """
    """
    def __init__(self, path: str, uuid: str):
        self.path = Path(path) / uuid
        self.path.mkdir(exist_ok=True, parents=True)

        self.chk_path = self.path / 'chk'
        self.chk_path.mkdir(exist_ok=True)

        self.log_path = self.path / 'logs'
        self.log_path.mkdir(exist_ok=True)

        self.params_path = self.path / 'params'
        self.params_path.mkdir(exist_ok=True)

        # TB logs
        self.writer = SummaryWriter(str(self.path))

        # Dump the `git log` and `git diff`. In this way one can checkout
        #  the last commit, add the diff and should be in the same state.
        for cmd in ['log', 'diff']:
            with open(self.path / f'git_{cmd}.txt', mode='wt') as f:
                subprocess.run(['git', cmd], stdout=f)

    def load_logs(self):
        with open(str(self.params_path / 'params.json'), 'r') as fp:
            params = json.load(fp)
        with open(str(self.params_path / 'hparams.json'), 'r') as fp:
            hparams = json.load(fp)
        return params, hparams

    @staticmethod
    def load_net(path: str, chk_name: str, dataset_name: str):
        with open(str(Path(path) / 'params' / 'hparams.json'), 'r') as fp:
            net_hparams = json.load(fp)
        with open(str(Path(path) / 'params' / 'params.json'), 'r') as fp:
            net_params = json.load(fp)

        assert dataset_name == net_params['dataset_name']
        net = TriNet(backbone_type=net_hparams['backbone_type'], pretrained=True,
                     num_classes=net_hparams['num_classes'])
        net_state_dict = torch.load(Path(path) / 'chk' / chk_name)
        net.load_state_dict(net_state_dict)
        return net

    def write_logs(self, model: torch.nn.Module, params: dict):
        with open(str(self.params_path / 'params.json'), 'w') as fp:
            json.dump(params, fp)
        with open(str(self.params_path / 'hparams.json'), 'w') as fp:
            json.dump(model.get_hparams(), fp)

    def write_image(self, image: np.ndarray, epoch: int, name: str):
        out_image_path = self.log_path / f'{epoch:05d}_{name}.jpg'
        cv2.imwrite(str(out_image_path), image)

        image = image[..., ::-1]
        self.writer.add_image(f'{name}', image, epoch, dataformats='HWC')

    def dump_metric_tb(self, value: float, epoch: int, m_type: str, m_desc: str):
        self.writer.add_scalar(f'{m_type}/{m_desc}', value, epoch)

    def save_net(self, net: torch.nn.Module, name: str = 'weights', overwrite: bool = False):
        weights_path = self.chk_path / name
        if weights_path.exists() and not overwrite:
            raise ValueError('PREVENT OVERWRITE WEIGHTS')
        torch.save(net.state_dict(), weights_path)

    def dump_hparams(self, hparams: dict, metrics: dict):
        self.writer.add_hparams(hparams, metrics)
