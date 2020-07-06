import os.path as osp
import numpy as np
from pathlib import Path
from copy import deepcopy

"""Dataset classes"""


class DukeVideoreID(object):
    """
    Duke VIDEO re-id
    Reference:
    Ristani et al. Performance measures and a data set for multi-target, multi-camera tracking
    """

    def __init__(self, root='/data/datasets/', min_seq_len=0):
        self.root = osp.join(root, 'DukeMTMC-VideoReID')
        self.train_name_path = osp.join(self.root, 'train')
        self.gallery_name_path = osp.join(self.root, 'gallery')
        self.query_name_path = osp.join(self.root, 'query')

        train_paths = self._get_paths(self.train_name_path)
        query_paths = self._get_paths(self.query_name_path)
        gallery_paths = self._get_paths(self.gallery_name_path)

        train, num_train_tracklets, num_train_pids, num_train_imgs = self._process_video(train_paths, re_label=True)
        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs = self._process_video(gallery_paths, re_label=False)
        query, num_query_tracklets, num_query_pids, num_query_imgs = self._process_video(query_paths, re_label=False)

        # query and gallery image are computed from first frames
        query_img = []
        for el in query:
            query_img.append((el[0][:1], el[1], el[2]))  # first image of gallery tracklet

        gallery_img = []
        for el in gallery:
            gallery_img.append((el[0][:1], el[1], el[2]))  # first image of gallery tracklet

        num_imgs_per_tracklet = num_train_imgs + num_gallery_imgs + num_query_imgs
        total_num = np.sum(num_imgs_per_tracklet)
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_gallery_tracklets + num_query_tracklets

        print("=> DUKE re-ID loaded")
        print("Dataset statistics:")
        print("  -----------------------------------------")
        print("  subset   | # ids | # tracklets | # images")
        print("  -----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:8d}".format(num_train_pids, num_train_tracklets, np.sum(num_train_imgs)))
        print("  query    | {:5d} | {:8d} | {:8d}".format(num_query_pids, num_query_tracklets, np.sum(num_query_imgs)))
        print("  gallery  | {:5d} | {:8d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets, np.sum(num_gallery_imgs)))
        print("  -----------------------------------------")
        print("  total    | {:5d} | {:8d} | {:8d}".format(num_total_pids, num_total_tracklets, total_num))
        print("  -----------------------------------------")
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  -----------------------------------------")

        self.train = train
        self.gallery = gallery
        self.query = query
        self.query_img = query_img
        self.gallery_img = gallery_img

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _get_paths(self, fpath):
        path = Path(fpath)
        return sorted(path.glob('**/*.jpg'))

    def _process_video(self, train_paths, re_label: bool):
        train = []
        pids = []
        num_images = []
        train_names = [p.name for p in train_paths]
        hs = np.asarray([hash(el[:7]) for el in train_names])
        displaces = np.nonzero(hs[1:] - hs[:-1])[0] + 1
        displaces = np.concatenate([[0], displaces, [len(hs)]])
        for idx in range(len(displaces) - 1):
            names = train_names[displaces[idx]: displaces[idx + 1]]
            pid, camera = map(int, names[0].replace('C', '').split('_')[:2])
            camera -= 1
            paths = [str(p) for p in train_paths[displaces[idx]: displaces[idx + 1]]]
            train.append((paths, pid, camera))
            num_images.append(len(names))
            pids.append(pid)

        # RE-LABEl
        if re_label:
            pid_map = {pid: idx for idx, pid in enumerate(np.unique(pids))}
            for i in range(len(train)):
                pid = pids[i]
                train[i] = (train[i][0], pid_map[pid], train[i][2])
                pids[i] = pid_map[pid]

        return train, len(train), len(set(pids)), num_images
