from argparse import Namespace

import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

from torchvision import transforms as T
import data.temporal_transforms as TT
from data.dukevideoreid.data_manager import DukeVideoreID
from data.mars.data_manager import Mars
from data.misc import get_default_video_loader
from data.misc import get_transforms
from data.samplers import ReIDBatchSampler
from data.veri.data_manager import Veri
from utils.misc import init_worker


class Dataset(data.Dataset):
    """Video ReID Dataset.
    Note:
        Batch data has shape N x C x T x H x W
    Args:
        dataset (list): List with items (img_paths, pid, camid)
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self,
                 dataset,
                 spatial_transform=None,
                 temporal_transform=None,
                 get_loader=get_default_video_loader):
        self.dataset = dataset
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()
        self.teacher_mode = False

    def __len__(self):
        return len(self.dataset)

    def get_num_pids(self):
        return len(np.unique([el[1] for el in self.dataset]))

    def get_num_cams(self):
        return len(np.unique([el[2] for el in self.dataset]))

    def set_teacher_mode(self, is_teacher: bool):
        self.teacher_mode = is_teacher

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (clip, pid, camid) where pid is identity of the clip.
        """
        img_paths, pid, camid = self.dataset[index]

        if isinstance(self.temporal_transform, TT.MultiViewTemporalTransform):
            candidates = list(filter(lambda x: x[1] == pid, self.dataset))
            img_paths = self.temporal_transform(candidates, index)
        elif self.temporal_transform is not None:
            img_paths = self.temporal_transform(img_paths, index)

        clip = self.loader(img_paths)

        if not self.teacher_mode:
            clip = [self.spatial_transform(img) for img in clip]
        else:
            clip_aug = [self.spatial_transform(img) for img in clip]
            std_daug = T.Compose([
                self.spatial_transform.transforms[0],
                T.ToTensor(),
                self.spatial_transform.transforms[-1] if not isinstance(self.spatial_transform.transforms[-1], T.RandomErasing) else self.spatial_transform.transforms[-2]
            ])
            clip_std = [std_daug(img) for img in clip]
            clip = clip_aug + clip_std

        clip = torch.stack(clip, 0)

        return clip, pid, camid


DATASETS = {
    'mars': Mars,
    'veri': Veri,
    'duke-video-reid': DukeVideoreID,
}


class DataConf:
    def __init__(self, perform_x2i, perform_x2v, augment_gallery):
        self.perform_x2i = perform_x2i
        self.perform_x2v = perform_x2v
        self.augment_gallery = augment_gallery


DATA_CONFS = {
    'mars': DataConf(perform_x2i=False, perform_x2v=True, augment_gallery=True),
    'veri': DataConf(perform_x2i=True, perform_x2v=True, augment_gallery=False),
    'duke-video-reid': DataConf(perform_x2i=False, perform_x2v=True, augment_gallery=False),

}


def get_dataloaders(dataset_name: str, root: str, device: torch.device, args: Namespace):
    dataset_name = dataset_name.lower()
    assert dataset_name in DATASETS.keys()
    dataset = DATASETS[dataset_name](root)

    pin_memory = True if device == torch.device('cuda') else False

    s_tr_train, t_tr_train = get_transforms(True, args)
    s_tr_test, t_tr_test = get_transforms(False, args)

    train_loader = DataLoader(
        Dataset(dataset.train, spatial_transform=s_tr_train,
                temporal_transform=t_tr_train),
        batch_sampler=ReIDBatchSampler(dataset.train, p=args.p, k=args.k),
        num_workers=args.workers, pin_memory=pin_memory,
        worker_init_fn=init_worker
    )

    query_loader = DataLoader(
        Dataset(dataset.query, spatial_transform=s_tr_test,
                temporal_transform=t_tr_test),
        batch_size=args.test_batch, shuffle=False, num_workers=2,
        pin_memory=pin_memory, drop_last=False, worker_init_fn=init_worker
    )

    gallery_loader = DataLoader(
        Dataset(dataset.gallery, spatial_transform=s_tr_test,
                temporal_transform=t_tr_test),
        batch_size=args.test_batch, shuffle=False, num_workers=2,
        pin_memory=pin_memory, drop_last=False, worker_init_fn=init_worker
    )

    queryimg_loader = DataLoader(
        Dataset(dataset.query_img, spatial_transform=s_tr_test),
        batch_size=args.img_test_batch, shuffle=False, num_workers=2,
        pin_memory=pin_memory, drop_last=False, worker_init_fn=init_worker
    )

    galleryimg_loader = DataLoader(
        Dataset(dataset.gallery_img, spatial_transform=s_tr_test),
        batch_size=args.img_test_batch, shuffle=False, num_workers=2,
        pin_memory=pin_memory, drop_last=False, worker_init_fn=init_worker
    )

    return train_loader, query_loader, gallery_loader, queryimg_loader, galleryimg_loader
