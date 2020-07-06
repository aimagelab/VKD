import os.path as osp
import numpy as np
from copy import deepcopy


class Veri(object):
    """
    VeRi

    Reference:
    """

    def __init__(self, root='/data/datasets/', min_seq_len=0):
        self.root = osp.join(root, 'VeRi')
        self.train_name_path = osp.join(self.root, 'name_train.txt')
        self.query_name_path = osp.join(self.root, 'name_query.txt')
        self.track_gallery_info_path = osp.join(self.root, 'test_track.txt')

        train_names = self._get_names(self.train_name_path)
        query_names = self._get_names(self.query_name_path)
        train, num_train_tracklets, num_train_pids, num_train_imgs = self._process_train(train_names)
        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs = self._process_gallery()
        query, num_query_tracklets, num_query_pids, num_query_imgs = self._process_query(query_names, gallery)

        gallery_img = []
        num_gallery_tracklets = 0
        for el in gallery:
            for fr in el[0]:
                gallery_img.append(([fr], el[1], el[2]))
                num_gallery_tracklets += 1

        query_img, _, _, _ = self._process_query_image(query_names)

        num_imgs_per_tracklet = num_train_imgs + num_gallery_imgs + num_query_imgs
        total_num = np.sum(num_imgs_per_tracklet)
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_gallery_tracklets + num_query_tracklets

        print("=> VeRi loaded")
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
        self.query = query
        self.gallery = gallery
        self.gallery_img = gallery_img
        self.query_img = query_img

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def _process_gallery(self):
        gallery = []
        pids = []
        num_images = []

        with open(self.track_gallery_info_path) as fp:
            for line in fp.readlines():
                imgs_names = [osp.join(self.root, f'image_test/{el}') for el in line.split(' ')]
                imgs_names = imgs_names[1:len(imgs_names)-1]
                pid, camera = map(int, line.split(' ')[0].replace('c', '').split('_')[:2])
                camera -= 1
                gallery.append((imgs_names, pid, camera))
                num_images.append(len(imgs_names))
                pids.append(pid)

        return gallery, len(gallery), len(set(pids)), num_images

    def _process_train(self, train_names):
        train = []
        pids = []
        num_images = []
        train_names = sorted(train_names)
        hs = np.asarray([hash(el[:9]) for el in train_names])
        displaces = np.nonzero(hs[1:] - hs[:-1])[0] + 1
        displaces = np.concatenate([[0], displaces, [len(hs)]])
        for idx in range(len(displaces) - 1):
            names = train_names[displaces[idx]: displaces[idx + 1]]
            imgs_names = [osp.join(self.root, f'image_train/{el}') for el in names]
            pid, camera = map(int, names[0].replace('c', '').split('_')[:2])
            camera -= 1
            train.append((imgs_names, pid, camera))
            num_images.append(len(imgs_names))
            pids.append(pid)

        # RE-LABEl
        pid_map = {pid: idx for idx, pid in enumerate(np.unique(pids))}
        for i in range(len(train)):
            pid = pids[i]
            train[i] = (train[i][0], pid_map[pid], train[i][2])
            pids[i] = pid_map[pid]

        return train, len(train), len(set(pids)), num_images

    def _process_query(self, query_names, gallery):
        queries = []
        pids = []
        num_images = []

        for qn in query_names:
            pid, camera = map(int, qn.replace('c', '').split('_')[:2])
            camera -= 1
            # look into gallery
            for el in gallery:
                if el[1] == pid and el[2] == camera:
                    queries.append((deepcopy(el[0]), el[1], el[2]))
                    num_images.append(len(el[0]))
                    pids.append(el[1])
                    break

        return queries, len(queries), len(set(pids)), num_images

    def _process_query_image(self, query_names):
        queries = []
        pids = []
        num_images = []

        for qn in query_names:
            imgs_names = [osp.join(self.root, f'image_query/{qn}')]
            pid, camera = map(int, qn.replace('c', '').split('_')[:2])
            camera -= 1
            queries.append((imgs_names, pid, camera))
            num_images.append(len(imgs_names))

            pids.append(pid)

        return queries, len(queries), len(set(pids)), num_images
