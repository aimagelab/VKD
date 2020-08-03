import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from tqdm import tqdm

from data.datasets import get_dataloaders
from utils.conf import Conf
from utils.saver import Saver

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from torch.nn.functional import adaptive_avg_pool2d
from torch.nn.functional import relu
from torch.nn.functional import normalize

from model.net import TriNet

class Hook:
    def __init__(self):
        self.buffer = []

    def __call__(self, module, _, ten_out):
        self.buffer.append(ten_out)

    def reset(self):
        self.buffer = []


def parse(conf: Conf):

    parser = argparse.ArgumentParser(description='Train img to video model')
    parser = conf.add_default_args(parser)

    parser.add_argument('net1',    type=str, help='Path to TriNet base folder.')
    parser.add_argument('--chk_net1',  type=str, help='checkpoint name', default='chk_end')
    parser.add_argument('net2', type=str, help='Path to TriNet base folder.')
    parser.add_argument('--chk_net2', type=str, help='checkpoint name', default='chk_end')
    parser.add_argument('--dest_path',  type=Path, default='/tmp/heatmaps_out')

    args = parser.parse_args()
    args.train_strategy = 'multiview'
    args.use_random_erasing = False
    args.num_train_images = 0
    args.img_test_batch = 32

    return args


def extract_grad_cam(net: TriNet, inputs: torch.Tensor, device: torch.device,
                     hook: Hook):

    _, logits = net(inputs, return_logits=True)  # forward calls hooks
    logits_max = torch.max(logits, 1)[0]

    conv_features = hook.buffer[0]

    grads = torch.autograd.grad(logits_max, conv_features,
                                grad_outputs=torch.ones(len(conv_features)).to(device))[0]

    with torch.no_grad():
        weights = adaptive_avg_pool2d(grads, (1, 1))
        attn = relu(torch.sum(conv_features * weights, 1))
        old_shape = attn.shape
        attn = normalize(attn.view(attn.shape[0], -1))
        attn = attn.view(old_shape)

    return attn.view(*inputs.shape[:2], *attn.shape[1:])


def save_img(img, attn, dest_path):
    height, width = img.shape[0], img.shape[1]
    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img, origin='upper')
    if attn is not None:
        ax.imshow(attn, origin='upper', extent=[0, width, height, 0],
                  alpha=0.4, cmap=plt.cm.get_cmap('jet'))
    fig.canvas.draw()
    plt.savefig(dest_path, dpi=height)
    plt.close()


def main():
    conf = Conf()
    conf.suppress_random()
    device = conf.get_device()

    args = parse(conf)

    dest_path = args.dest_path / (Path(args.net1).name + '__vs__' + Path(args.net2).name)
    dest_path.mkdir(exist_ok=True, parents=True)

    both_path = dest_path / 'both'
    both_path.mkdir(exist_ok=True, parents=True)

    net1_path = dest_path / Path(args.net1).name
    net1_path.mkdir(exist_ok=True, parents=True)

    net2_path = dest_path / Path(args.net2).name
    net2_path.mkdir(exist_ok=True, parents=True)

    orig_path = dest_path / 'orig'
    orig_path.mkdir(exist_ok=True, parents=True)

    # ---- Restore net
    net1 = Saver.load_net(args.net1, args.chk_net1, args.dataset_name).to(device)
    net2 = Saver.load_net(args.net2, args.chk_net2, args.dataset_name).to(device)

    net1.eval()
    net2.eval()

    train_loader, query_loader, gallery_loader, queryimg_loader, galleryimg_loader = \
        get_dataloaders(args.dataset_name, conf.nas_path, device, args)

    # register hooks
    hook_net_1, hook_net_2 = Hook(), Hook()

    net1.backbone.features_layers[4].register_forward_hook(hook_net_1)
    net2.backbone.features_layers[4].register_forward_hook(hook_net_2)

    dst_idx = 0

    for idx_batch, (vids, *_) in enumerate(tqdm(galleryimg_loader, 'iterating..')):
        if idx_batch < len(galleryimg_loader) - 50:
            continue
        net1.zero_grad()
        net2.zero_grad()

        hook_net_1.reset()
        hook_net_2.reset()

        vids = vids.to(device)
        attn_1 = extract_grad_cam(net1, vids, device, hook_net_1)
        attn_2 = extract_grad_cam(net2, vids, device, hook_net_2)

        B, N_VIEWS = attn_1.shape[0], attn_1.shape[1]

        for idx_b in range(B):
            for idx_v in range(N_VIEWS):

                el_img = vids[idx_b, idx_v]
                el_attn_1 = attn_1[idx_b, idx_v]
                el_attn_2 = attn_2[idx_b, idx_v]

                el_img = el_img.cpu().numpy().transpose(1, 2, 0)
                el_attn_1 = el_attn_1.cpu().numpy()
                el_attn_2 = el_attn_2.cpu().numpy()

                mean, var = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                el_img = (el_img * var) + mean
                el_img = np.clip(el_img, 0, 1)

                el_attn_1 = cv2.blur(el_attn_1, (3, 3))
                el_attn_1 = cv2.resize(el_attn_1, (el_img.shape[1], el_img.shape[0]),
                                       interpolation=cv2.INTER_CUBIC)

                el_attn_2 = cv2.blur(el_attn_2, (3, 3))
                el_attn_2 = cv2.resize(el_attn_2, (el_img.shape[1], el_img.shape[0]),
                                       interpolation=cv2.INTER_CUBIC)

                save_img(el_img, el_attn_1, net1_path / f'{dst_idx}.png')
                save_img(el_img, el_attn_2, net2_path / f'{dst_idx}.png')

                save_img(el_img, None, orig_path / f'{dst_idx}.png')

                save_img(np.concatenate([el_img, el_img], 1),
                         np.concatenate([el_attn_1, el_attn_2], 1), both_path / f'{dst_idx}.png')

                dst_idx += 1


if __name__ == '__main__':
    main()

