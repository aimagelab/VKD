import argparse
from copy import deepcopy
from uuid import uuid4

import torch
from torch import nn
from torch.optim import Adam
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from data.datasets import get_dataloaders, DATA_CONFS
from model.loss import KDLoss
from model.loss import OnlineTripletLoss
from model.loss import SimilarityDistillationLoss
from model.loss import LogitsMatching
from model.net import TriNet
from tools.eval import Evaluator
from utils.conf import Conf
from utils.misc import AvgMeter
from utils.misc import str2bool
from utils.saver import Saver


class LearningRateGenDecayer(object):

    def __init__(self, initial_lr: float, decay: float, min: float = 1e-5):
        self.decay = decay
        self.initial_lr = initial_lr

    def __call__(self, epoch: int):
        return max(self.initial_lr * (self.decay ** epoch), 1e-5)


def parse(conf: Conf):
    parser = argparse.ArgumentParser(description='Train img to video model')
    parser = conf.add_default_args(parser)

    parser.add_argument('teacher', type=str)
    parser.add_argument('--teacher_chk_name', type=str, default='chk_end')

    parser.add_argument('--student', type=str)
    parser.add_argument('--student_chk_name', type=str, default='chk_end')

    parser.add_argument('--exp_name', type=str, default=str(uuid4()))
    parser.add_argument('--num_generations', type=int, default=1)

    parser.add_argument('--eval_epoch_interval', type=int, default=50)
    parser.add_argument('--print_epoch_interval', type=int, default=5)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--temp', type=float, default=10.)
    parser.add_argument('--lambda_coeff', type=float, default=0.0001)
    parser.add_argument('--kl_coeff', type=float, default=0.1)

    parser.add_argument('--num_train_images', type=int, default=8)
    parser.add_argument('--num_student_images', type=int, default=2)

    parser.add_argument('--train_strategy', type=str, default='multiview',
                        choices=['multiview', 'temporal'])

    parser.add_argument('--num_epochs', type=int, default=400)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--first_milestone', type=int, default=300)
    parser.add_argument('--step_milestone', type=int, default=50)

    parser.add_argument('--reinit_l4', type=str2bool, default=True)
    parser.add_argument('--reinit_l3', type=str2bool, default=False)

    parser.add_argument('--logits_dist', type=str, default='kl',
                        choices=['kl', 'mse'])

    args = parser.parse_args()
    args.use_random_erasing = True

    return args


class DistillationTrainer:

    def __init__(self, train_loader: DataLoader, query_loader: DataLoader,
                 gallery_loader: DataLoader, queryimg_loader: DataLoader,
                 galleryimg_loader: DataLoader, device: torch.device, saver: Saver,
                 args: argparse.Namespace, conf: Conf):

        self.class_loss = nn.CrossEntropyLoss(reduction='mean').to(device)
        self.distill_loss = KDLoss(temp=args.temp, reduction='mean').to(device) \
            if args.logits_dist == 'kl' else LogitsMatching(reduction='mean')
        self.similarity_loss = SimilarityDistillationLoss(metric='l2').to(device)
        self.triplet_loss = OnlineTripletLoss('soft', True, reduction='mean').to(device)

        self.train_loader = train_loader
        self.query_loader = query_loader
        self.gallery_loader = gallery_loader
        self.queryimg_loader = queryimg_loader
        self.galleryimg_loader = galleryimg_loader

        self.device = device
        self.saver = saver
        self.args = args
        self.conf = conf

        self.lr = LearningRateGenDecayer(initial_lr=self.args.lr,
                                         decay=self.args.lr_decay)
        self._epoch = 0
        self._gen = 0

    def evaluate(self, net: nn.Module):
        ev = Evaluator(net, query_loader, gallery_loader, queryimg_loader, galleryimg_loader,
                       DATA_CONFS[self.args.dataset_name], device)
        ev.eval(self.saver, self._epoch, self.args.eval_epoch_interval, self.args.verbose)

    def __call__(self, teacher_net: TriNet, student_net: TriNet):

        opt = Adam(student_net.parameters(), lr=self.lr(self._gen), weight_decay=1e-5)

        milestones = list(range(self.args.first_milestone, self.args.num_epochs,
                                self.args.step_milestone))

        scheduler = lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=self.args.gamma)

        for e in range(self.args.num_epochs):

            if e % self.args.eval_epoch_interval == 0 and e > 0:
                self.evaluate(student_net)

            avm = AvgMeter(['kl', 'triplet', 'class', 'similarity', 'loss'])

            student_net.student_mode()
            teacher_net.teacher_mode()

            for x, y, cams in self.train_loader:

                x, y = x.to(self.device), y.to(self.device)
                x_ = torch.stack([x[i, torch.randperm(x.shape[1])] for i in range(x.shape[0])])

                x_teacher, x_student = x, x_[:, :self.args.num_student_images]

                with torch.no_grad():
                    teacher_emb, teacher_logits = teacher_net(x_teacher, return_logits=True)

                opt.zero_grad()

                student_emb, student_logits = student_net(x_student, return_logits=True)

                kl_div_batch = self.distill_loss(teacher_logits, student_logits)
                similarity_loss_batch = self.similarity_loss(teacher_emb, student_emb)
                triplet_loss_batch = self.triplet_loss(student_emb, y)
                class_loss_batch = self.class_loss(student_logits, y)

                loss = (triplet_loss_batch + class_loss_batch) + \
                       self.args.lambda_coeff * (similarity_loss_batch) + \
                       self.args.kl_coeff * (kl_div_batch)

                avm.add([kl_div_batch.item(), triplet_loss_batch.item(),
                         class_loss_batch.item(), similarity_loss_batch.item(),
                         loss.item()])

                loss.backward()
                opt.step()

            scheduler.step()

            if self._epoch % self.args.print_epoch_interval == 0:
                stats = avm()
                str_ = f"Epoch: {self._epoch}"
                for (l, m) in stats:
                    str_ += f" - {l} {m:.2f}"
                    self.saver.dump_metric_tb(m, self._epoch, 'losses', f"avg_{l}")
                self.saver.dump_metric_tb(opt.defaults['lr'], self._epoch, 'lr', 'lr')
                print(str_)

            self._epoch += 1

        self._gen += 1

        return student_net


if __name__ == '__main__':
    conf = Conf()
    device = conf.get_device()
    args = parse(conf)

    conf.suppress_random(set_determinism=args.set_determinism)

    train_loader, query_loader, gallery_loader, queryimg_loader, galleryimg_loader = \
        get_dataloaders(args.dataset_name, conf.nas_path, device, args)

    teacher_net: TriNet = Saver.load_net(args.teacher,
                                         args.teacher_chk_name, args.dataset_name).to(device)

    student_net: TriNet = deepcopy(teacher_net) if args.student is None \
        else Saver.load_net(args.student, args.student_chk_name, args.dataset_name)
    student_net = student_net.to(device)

    ev = Evaluator(student_net, query_loader, gallery_loader, queryimg_loader, galleryimg_loader,
                   DATA_CONFS[args.dataset_name], device)

    print('v' * 100)
    ev.eval(saver=None, iteration=None, verbose=True, do_tb=False)
    print('v' * 100)

    student_net.reinit_layers(args.reinit_l4, args.reinit_l3)

    saver = Saver(conf.log_path, args.exp_name)
    saver.write_logs(student_net, vars(args))

    d_trainer: DistillationTrainer = DistillationTrainer(train_loader, query_loader,
                                   gallery_loader, queryimg_loader, galleryimg_loader, conf.get_device(),
                                   saver, args, conf)

    print("EXP_NAME: ", args.exp_name)

    for idx_iteration in range(args.num_generations):
        print(f'starting generation {idx_iteration+1}')
        print('#'*100)
        teacher_net = d_trainer(teacher_net, student_net)
        d_trainer.evaluate(teacher_net)
        teacher_net.teacher_mode()

        student_net = deepcopy(teacher_net)
        saver.save_net(student_net, f'chk_di_{idx_iteration + 1}')

        student_net.reinit_layers(args.reinit_l4, args.reinit_l3)

    saver.writer.close()
