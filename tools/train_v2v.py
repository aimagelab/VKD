import argparse
from uuid import uuid4

from torch import nn
from torch.optim import Adam
from torch.optim import lr_scheduler

from data.datasets import get_dataloaders, DATA_CONFS
from model.loss import OnlineTripletLoss
from model.net import get_model
from tools.eval import Evaluator
from utils.conf import Conf
from utils.saver import Saver
from utils.misc import str2bool

from utils.misc import AvgMeter


def parse(conf: Conf):

    parser = argparse.ArgumentParser(description='Train img to video model')
    parser = conf.add_default_args(parser)

    parser.add_argument('--exp_name',   type=str, default=str(uuid4()), help='Experiment name.')
    parser.add_argument('--metric',   type=str, default='euclidean',
                        choices=['euclidean', 'cosine'], help='Metric for distances')
    parser.add_argument('--num_train_images', type=int, default=8, help='Num. of bag images.')

    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--eval_epoch_interval', type=int, default=50)
    parser.add_argument('--save_epoch_interval', type=int, default=50)
    parser.add_argument('--print_epoch_interval', type=int, default=5)

    parser.add_argument('--wd', type=float, default=1e-5)

    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--first_milestone', type=int, default=200)
    parser.add_argument('--step_milestone', type=int, default=50)

    parser.add_argument('--use_random_erasing', type=str2bool, default=True)
    parser.add_argument('--train_strategy', type=str, default='chunk',
                        choices=['multiview', 'chunk'])

    args = parser.parse_args()
    return args


def main():
    conf = Conf()
    args = parse(conf)
    device = conf.get_device()

    conf.suppress_random(set_determinism=args.set_determinism)
    saver = Saver(conf.log_path, args.exp_name)

    train_loader, query_loader, gallery_loader, queryimg_loader, galleryimg_loader = \
        get_dataloaders(args.dataset_name, conf.nas_path, device, args)

    num_pids = train_loader.dataset.get_num_pids()

    net = nn.DataParallel(get_model(args, num_pids))
    net = net.to(device)

    saver.write_logs(net.module, vars(args))

    opt = Adam(net.parameters(), lr=1e-4, weight_decay=args.wd)
    milestones = list(range(args.first_milestone, args.num_epochs,
                            args.step_milestone))
    scheduler = lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=args.gamma)

    triplet_loss = OnlineTripletLoss('soft', True, reduction='mean').to(device)
    class_loss = nn.CrossEntropyLoss(reduction='mean').to(device)

    print("EXP_NAME: ", args.exp_name)

    for e in range(args.num_epochs):

        if e % args.eval_epoch_interval == 0 and e > 0:
            ev = Evaluator(net, query_loader, gallery_loader, queryimg_loader, galleryimg_loader,
                           DATA_CONFS[args.dataset_name], device)
            ev.eval(saver, e, args.verbose)

        if e % args.save_epoch_interval == 0 and e > 0:
            saver.save_net(net.module, f'chk_{e // args.save_epoch_interval}')

        avm = AvgMeter(['triplet', 'class'])

        for it, (x, y, cams) in enumerate(train_loader):
            net.train()

            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            embeddings, f_class = net(x, return_logits=True)

            triplet_loss_batch = triplet_loss(embeddings, y)
            class_loss_batch = class_loss(f_class, y)
            loss = triplet_loss_batch + class_loss_batch

            avm.add([triplet_loss_batch.item(), class_loss_batch.item()])

            loss.backward()
            opt.step()

        if e % args.print_epoch_interval == 0:
            stats = avm()
            str_ = f"Epoch: {e}"
            for (l, m) in stats:
                str_ += f" - {l} {m:.2f}"
                saver.dump_metric_tb(m, e, 'losses', f"avg_{l}")
            saver.dump_metric_tb(opt.param_groups[0]['lr'], e, 'lr', 'lr')
            print(str_)

        scheduler.step()

    ev = Evaluator(net, query_loader, gallery_loader, queryimg_loader, galleryimg_loader,
                   DATA_CONFS[args.dataset_name], device)
    ev.eval(saver, e, args.verbose)

    saver.save_net(net.module, 'chk_end')
    saver.writer.close()


if __name__ == '__main__':
    main()
