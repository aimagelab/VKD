import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.datasets import get_dataloaders
from data.eval_metrics import evaluate
from model.net import get_model
from utils.conf import Conf
from utils.saver import Saver
from data.datasets import DataConf, DATA_CONFS


class Evaluator:

    def __init__(self, model: torch.nn.Module, query_loader: DataLoader,
                 gallery_loader: DataLoader, queryimg_loader: DataLoader,
                 galleryimg_loader: DataLoader, data_conf: DataConf,
                 device: torch.device):

        self.perform_x2i = data_conf.perform_x2i
        self.perform_x2v = data_conf.perform_x2v
        model.eval()

        self.gallery_loader = gallery_loader
        self.query_loader = query_loader
        self.queryimg_loader = queryimg_loader
        self.galleryimg_loader = galleryimg_loader

        # ----------- QUERY
        vid_qf, self.vid_q_pids, self.vid_q_camids = self.extract_features(model, query_loader,
                                                                           device)
        img_qf, self.img_q_pids, self.img_q_camids = self.extract_features(model, queryimg_loader,
                                                                           device)
        # ----------- GALLERY
        if self.perform_x2v:
            vid_gf, self.vid_g_pids, self.vid_g_camids = self.extract_features(model,
                                                                               gallery_loader,
                                                                               device)
        if self.perform_x2i:
            img_gf, self.img_g_pids, self.img_g_camids = self.extract_features(model,
                                                                               galleryimg_loader,
                                                                               device)

        if data_conf.augment_gallery:
            # gallery must contain query, if not 140 query will not have ground truth in MARS.
            if self.perform_x2v:
                vid_gf = torch.cat((vid_qf, vid_gf), 0)
                self.vid_g_pids = np.append(self.vid_q_pids, self.vid_g_pids)
                self.vid_g_camids = np.append(self.vid_q_camids, self.vid_g_camids)
            if self.perform_x2i:
                img_gf = torch.cat((img_qf, img_gf), 0)
                self.img_g_pids = np.append(self.img_q_pids, self.img_g_pids)
                self.img_g_camids = np.append(self.img_q_camids, self.img_g_camids)

        if self.perform_x2v:
            self.v2v_distmat = self.compute_distance_matrix(vid_qf, vid_gf, metric='cosine').numpy()
            self.i2v_distmat = self.compute_distance_matrix(img_qf, vid_gf, metric='cosine').numpy()
        if self.perform_x2i:
            self.v2i_distmat = self.compute_distance_matrix(vid_qf, img_gf, metric='cosine').numpy()
            self.i2i_distmat = self.compute_distance_matrix(img_qf, img_gf, metric='cosine').numpy()

    @staticmethod
    def compute_distance_matrix(x: torch.Tensor, y: torch.Tensor, metric='cosine'):
        if metric == 'cosine':
            x = x / torch.norm(x, dim=-1, keepdim=True)
            y = y / torch.norm(y, dim=-1, keepdim=True)

        return 1 - torch.mm(x, y.T)

    def evaluate_v2v(self, verbose: bool = True):
        cmc, mAP, wrong_matches = evaluate(self.v2v_distmat, self.vid_q_pids,
                                           self.vid_g_pids, self.vid_q_camids,
                                           self.vid_g_camids, True)

        if verbose:
            print(f'V2V')
            print(f'top1:{cmc[0]:.2%} top5:{cmc[4]:.2%} top10:{cmc[9]:.2%} mAP:{mAP:.2%}')

        return cmc, mAP

    def evaluate_i2v(self, verbose: bool = True):

        cmc, mAP, wrong_matches = evaluate(self.i2v_distmat, self.img_q_pids,
                                           self.vid_g_pids, self.img_q_camids,
                                           self.vid_g_camids, True)
        if verbose:
            print(f'I2V')
            print(f'top1:{cmc[0]:.2%} top5:{cmc[4]:.2%} top10:{cmc[9]:.2%} mAP:{mAP:.2%}')

        return cmc, mAP

    def evaluate_v2i(self, verbose: bool = True):

        cmc, mAP, wrong_matches = evaluate(self.v2i_distmat, self.vid_q_pids,
                                           self.img_g_pids, self.vid_q_camids,
                                           self.img_g_camids, True)

        if verbose:
            print(f'V21')
            print(f'top1:{cmc[0]:.2%} top5:{cmc[4]:.2%} top10:{cmc[9]:.2%} mAP:{mAP:.2%}')

        return cmc, mAP

    def evaluate_i2i(self, verbose: bool = True):

        cmc, mAP, wrong_matches = evaluate(self.i2i_distmat, self.img_q_pids,
                                           self.img_g_pids, self.img_q_camids,
                                           self.img_g_camids, True)
        if verbose:
            print(f'I2I')
            print(f'top1:{cmc[0]:.2%} top5:{cmc[4]:.2%} top10:{cmc[9]:.2%} mAP:{mAP:.2%}')

        return cmc, mAP

    @torch.no_grad()
    def extract_features(self, model: torch.nn.Module, loader: DataLoader,
                         device: torch.device):
        """
        Extract features for the entire dataloader. It returns also pids and cams.
        """
        features, pids, cams = [], [], []

        for vids, pidids, camids in loader:
            vids = vids.to(device)

            feat = model(vids)
            feat = feat.data

            features.append(feat)
            pids.extend(pidids)
            cams.extend(camids)

        features = torch.cat(features, 0).to('cpu')
        pids = np.asarray(pids)
        cams = np.asarray(cams)
        return features, pids, cams

    @staticmethod
    def tb_cmc(saver: Saver, cmc_scores, it, method):
        for cmc_v in [0, 4, 9]:
            saver.dump_metric_tb(cmc_scores[cmc_v], it,
                                 f'{method}', f'cmc{cmc_v + 1}')

    def eval(self, saver: Saver, iteration: int, verbose: bool, do_tb: bool = True):

        if self.perform_x2v:
            cmc_scores_i2v, mAP_i2v = self.evaluate_i2v(verbose=verbose)
            if do_tb:
                saver.dump_metric_tb(mAP_i2v, iteration, 'i2v', f'mAP')
                self.tb_cmc(saver, cmc_scores_i2v, iteration, 'i2v')

            cmc_scores_v2v, mAP_v2v = self.evaluate_v2v(verbose=verbose)
            if do_tb:
                saver.dump_metric_tb(mAP_v2v, iteration, 'v2v', f'mAP')
                self.tb_cmc(saver, cmc_scores_v2v, iteration, 'v2v')

        if self.perform_x2i:
            cmc_scores_i2i, mAP_i2i = self.evaluate_i2i(verbose=verbose)
            if do_tb:
                saver.dump_metric_tb(mAP_i2i, iteration, 'i2i', f'mAP')
                self.tb_cmc(saver, cmc_scores_i2i, iteration, 'i2i')

            cmc_scores_v2i, mAP_v2i = self.evaluate_v2i(verbose=verbose)
            if do_tb:
                saver.dump_metric_tb(mAP_v2i, iteration, 'v2i', f'mAP')
                self.tb_cmc(saver, cmc_scores_v2i, iteration, 'v2i')


def parse(conf: Conf):

    parser = argparse.ArgumentParser(description='Train img to video model')
    parser = conf.add_default_args(parser)

    parser.add_argument('trinet_folder',    type=str, help='Path to TriNet base folder.')
    parser.add_argument('--trinet_chk_name',  type=str, help='checkpoint name', default='chk_end')

    args = parser.parse_args()
    args.train_strategy = 'chunk'
    args.use_random_erasing = False
    args.num_train_images = 0

    return args


def main():
    conf = Conf()
    conf.suppress_random()
    device = conf.get_device()

    args = parse(conf)

    # ---- SAVER OLD NET TO RESTORE PARAMS
    saver_trinet = Saver(Path(args.trinet_folder).parent, Path(args.trinet_folder).name)
    old_params, old_hparams = saver_trinet.load_logs()
    args.backbone = old_params['backbone']
    args.metric = old_params['metric']

    train_loader, query_loader, gallery_loader, queryimg_loader, galleryimg_loader = \
        get_dataloaders(args.dataset_name, conf.nas_path, device, args)
    num_pids = train_loader.dataset.get_num_pids()

    assert num_pids == old_hparams['num_classes']

    net = get_model(args, num_pids).to(device)
    state_dict = torch.load(Path(args.trinet_folder) / 'chk' / args.trinet_chk_name)
    net.load_state_dict(state_dict)

    e = Evaluator(net, query_loader, gallery_loader, queryimg_loader, galleryimg_loader,
                  device=device, data_conf=DATA_CONFS[args.dataset_name])

    e.eval(None, 0, verbose=True, do_tb=False)


if __name__ == '__main__':
    main()
