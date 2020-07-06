import torch
from torch import nn as nn
from torch.nn import functional as F


def l2_norm(x: torch.Tensor):
    return x / torch.norm(x, dim=-1, keepdim=True)


class MatrixPairwiseDistances(nn.Module):

    def __init__(self):
        super(MatrixPairwiseDistances, self).__init__()

    def __call__(self, x: torch.Tensor, y: torch.Tensor = None):
        if y is not None:  # exact form of squared distances
            differences = x.unsqueeze(1) - y.unsqueeze(0)
        else:
            differences = x.unsqueeze(1) - x.unsqueeze(0)
        distances = torch.sum(differences * differences, -1)
        return distances


class SmoothedCCE(nn.Module):
    def __init__(self, num_classes: int, eps: float = 0.1, reduction: str='sum'):
        super(SmoothedCCE, self).__init__()
        self.reduction = reduction
        assert reduction in ['sum', 'mean']
        self.eps = eps
        self.num_classes = num_classes

        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.factor_0 = self.eps / self.num_classes
        self.factor_1 = 1 - ((self.num_classes - 1) / self.num_classes) * self.eps

    def labels_to_one_hot(self, labels):
        onehot = torch.ones(len(labels), self.num_classes).to(labels.device) * self.factor_0
        onehot[torch.arange(0, len(labels), dtype=torch.long), labels.long()] = self.factor_1
        return onehot

    def forward(self, feats, target):
        """
        target are long in [0, num_classes)!
        """
        one_hots = self.labels_to_one_hot(target)
        if self.reduction == 'sum':
            loss = torch.sum(-torch.sum(one_hots * self.logsoftmax(feats), -1))
        else:
            loss = torch.mean(-torch.sum(one_hots * self.logsoftmax(feats), -1))

        return loss

    def __call__(self, *args, **kwargs):
        return super(SmoothedCCE, self).__call__(*args, **kwargs)


class KDLoss(nn.Module):

    def __init__(self, temp: float, reduction: str):
        super(KDLoss, self).__init__()

        self.temp = temp
        self.reduction = reduction
        self.kl_loss = nn.KLDivLoss(reduction=reduction)

    def forward(self, teacher_logits: torch.Tensor, student_logits: torch.Tensor):

        student_softmax = F.log_softmax(student_logits / self.temp, dim=-1)
        teacher_softmax = F.softmax(teacher_logits / self.temp, dim=-1)

        kl = nn.KLDivLoss(reduction='none')(student_softmax, teacher_softmax)
        kl = kl.sum() if self.reduction == 'sum' else kl.sum(1).mean()
        kl = kl * (self.temp ** 2)

        return kl

    def __call__(self, *args, **kwargs):
        return super(KDLoss, self).__call__(*args, **kwargs)


class LogitsMatching(nn.Module):

    def __init__(self, reduction: str):
        super(LogitsMatching, self).__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self, teacher_logits: torch.Tensor, student_logits: torch.Tensor):
        return self.mse_loss(student_logits, teacher_logits)

    def __call__(self, *args, **kwargs):
        return super(LogitsMatching, self).__call__(*args, **kwargs)


class SimilarityDistillationLoss(nn.Module):

    def __init__(self, metric: str):
        assert metric in ['l2', 'l1', 'huber']
        super(SimilarityDistillationLoss, self).__init__()
        self.distances = MatrixPairwiseDistances()
        self.metric = metric

    def forward(self, teacher_embs: torch.Tensor, student_embs: torch.Tensor):
        teacher_distances = self.distances(teacher_embs)
        student_distances = self.distances(student_embs)

        if self.metric == 'l2':
            return 0.5 * nn.MSELoss(reduction='mean')(student_distances, teacher_distances)
        if self.metric == 'l1':
            return 0.5 * nn.L1Loss(reduction='mean')(student_distances, teacher_distances)
        if self.metric == 'huber':
            return 0.5 * nn.SmoothL1Loss(reduction='mean')(student_distances, teacher_distances)
        raise ValueError()

    def __call__(self, *args, **kwargs):
        return super(SimilarityDistillationLoss, self).__call__(*args, **kwargs)


class OnlineTripletLoss(nn.Module):

    def __init__(self, margin='soft', batch_hard=True, reduction='mean'):
        super(OnlineTripletLoss, self).__init__()
        self.batch_hard = batch_hard
        self.reduction = reduction
        if isinstance(margin, float) or margin == 'soft':
            self.margin = margin
        else:
            raise NotImplementedError(
                'The margin {} is not recognized in TripletLoss()'.format(margin))

    def forward(self, feat, id=None, pos_mask=None, neg_mask=None, mode='id', dis_func='eu',
                n_dis=0):

        if dis_func == 'cdist':
            feat = feat / feat.norm(p=2, dim=1, keepdim=True)
            dist = self.cdist(feat, feat)
        elif dis_func == 'eu':
            dist = self.cdist(feat, feat)

        if mode == 'id':
            if id is None:
                raise RuntimeError('foward is in id mode, please input id!')
            else:
                identity_mask = torch.eye(feat.size(0)).byte()
                identity_mask = identity_mask.cuda() if id.is_cuda else identity_mask
                same_id_mask = torch.eq(id.unsqueeze(1), id.unsqueeze(0))
                negative_mask = same_id_mask ^ 1
                positive_mask = same_id_mask ^ identity_mask.bool()
        elif mode == 'mask':
            if pos_mask is None or neg_mask is None:
                raise RuntimeError('foward is in mask mode, please input pos_mask & neg_mask!')
            else:
                positive_mask = pos_mask
                same_id_mask = neg_mask ^ 1
                negative_mask = neg_mask
        else:
            raise ValueError('unrecognized mode')

        if self.batch_hard:
            if n_dis != 0:
                img_dist = dist[:-n_dis, :-n_dis]
                max_positive = (img_dist * positive_mask[:-n_dis, :-n_dis].float()).max(1)[0]
                min_negative = (img_dist + 1e5 * same_id_mask[:-n_dis, :-n_dis].float()).min(1)[0]
                dis_min_negative = dist[:-n_dis, -n_dis:].min(1)[0]
                z_origin = max_positive - min_negative
                # z_dis = max_positive - dis_min_negative
            else:
                max_positive = (dist * positive_mask.float()).max(1)[0]
                min_negative = (dist + 1e5 * same_id_mask.float()).min(1)[0]
                z = max_positive - min_negative
        else:
            pos = positive_mask.topk(k=1, dim=1)[1].view(-1, 1)
            positive = torch.gather(dist, dim=1, index=pos)
            pos = negative_mask.topk(k=1, dim=1)[1].view(-1, 1)
            negative = torch.gather(dist, dim=1, index=pos)
            z = positive - negative

        if isinstance(self.margin, float):
            b_loss = torch.clamp(z + self.margin, min=0)
        elif self.margin == 'soft':
            if n_dis != 0:
                b_loss = torch.log(1 + torch.exp(
                    z_origin)) + -0.5 * dis_min_negative  # + torch.log(1+torch.exp(z_dis))
            else:
                b_loss = torch.log(1 + torch.exp(z))
        else:
            raise NotImplementedError("How do you even get here!")

        if self.reduction == 'mean':
            return b_loss.mean()

        return b_loss.sum()

    def cdist(self, a, b):
        '''
        Returns euclidean distance between a and b

        Args:
             a (2D Tensor): A batch of vectors shaped (B1, D)
             b (2D Tensor): A batch of vectors shaped (B2, D)
        Returns:
             A matrix of all pairwise distance between all vectors in a and b,
             will be shape of (B1, B2)
        '''
        diff = a.unsqueeze(1) - b.unsqueeze(0)
        return ((diff ** 2).sum(2) + 1e-12).sqrt()

    def __call__(self, *args, **kwargs):
        return super(OnlineTripletLoss, self).__call__(*args, **kwargs)
