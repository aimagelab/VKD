import socket
import tempfile
import torch

from argparse import ArgumentParser

from model.net import BACKBONES
from model.net import Backbone

from utils.misc import str2bool
from data.datasets import DATASETS


class Conf:
    """
    This class encapsulate some configuration variables useful to have around.
    """

    SEED = 1897

    def __init__(self):
        """ Constructor class. """

        self.host_name = socket.gethostname()
        self.nas_path = self._get_nas_path()
        self.log_path = self._get_log_path()

    @staticmethod
    def add_default_args(parser: ArgumentParser):
        parser.add_argument('dataset_name', choices=list(DATASETS.keys()), type=str, help='dataset name')
        # Network
        parser.add_argument('--backbone', type=str, choices=BACKBONES, default=Backbone.RESNET_50,
                            help='Backbone network type.')
        parser.add_argument('--pretrained', type=str2bool, default=True, help='No pretraining.')

        # Others
        parser.add_argument('--set_determinism', type=str2bool, default=False)
        parser.add_argument('--test_batch', default=32, type=int)
        parser.add_argument('--img_test_batch', default=512, type=int)
        parser.add_argument('--verbose', type=str2bool, default=True, help='Debug mode')
        parser.add_argument('-j', '--workers', default=4, type=int)
        parser.add_argument('--p', type=int, default=18, help='')
        parser.add_argument('--k', type=int, default=4, help='')

        parser.add_argument('--num_test_images', type=int, default=8)

        return parser

    @staticmethod
    def get_tmp_path():
        return tempfile.gettempdir()

    @staticmethod
    def get_hostname_config():

        default_config = {
            'log_path': './logs',
            'nas_path': './datasets'
        }

        return default_config

    def _get_log_path(self) -> str:

        default_config = self.get_hostname_config()
        return default_config["log_path"]

    def _get_nas_path(self) -> str:

        default_config = self.get_hostname_config()
        return default_config["nas_path"]

    @staticmethod
    def suppress_random(seed: int = SEED, set_determinism: bool = False):
        import random
        import torch
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if set_determinism:
            torch.backends.cudnn.deterministic = True

    @staticmethod
    def get_device():
        return torch.device('cuda') if torch.cuda.is_available() \
            else torch.device('cpu')
