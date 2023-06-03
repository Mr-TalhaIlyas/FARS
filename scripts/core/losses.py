import yaml
with open('config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse

def one_hot(index, classes):
    
    size = index.size()[:1] + (classes,) # jsut gets shape -> (P, C)
    view = index.size()[:1] + (1,) # get shapes -> (P, 1)
    
    # makes a tensor of size (P, C) and fills it with zeros
    mask = torch.Tensor(size).fill_(0).to('cuda' if torch.cuda.is_available() else 'cpu') # (P, C)
    # reshapes the targets/index to (P, 1)
    index = index.view(view) 
    ones = 1.

    return mask.scatter_(1, index, ones) # places ones at those indexes (in mask) indicated by values in targets 


class FocalLoss(nn.Module):

    def __init__(self, gamma=2, eps=1e-7, size_average=True, one_hot=True, ignore=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.size_average = size_average
        self.one_hot = one_hot
        self.ignore = ignore

    def forward(self, input, target):
        '''
        0th index is ignored.
        '''
        B, C, H, W = input.size()
        input = input.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C => (P, C) 
        target = target.view(-1) # (P,)
        
        if self.ignore is not None:
            valid = (target != self.ignore)
            input = input[valid]
            target = target[valid]

        if self.one_hot:
            target = one_hot(target, C) # (P, C)

        probs = F.softmax(input, dim=1)
        probs = (probs * target).sum(1)
        probs = probs.clamp(self.eps, 1. - self.eps)

        log_p = probs.log()

        batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class Entropy(nn.Module):
    def __init__(self, ita=2.0, charbonnier=False, reduce_dim=False):
        super(Entropy, self).__init__()
        self.ita = ita
        self.charbonnier = charbonnier
        self.reduce_dim = reduce_dim

    def forward(self, inputs):

        _, C, _, _ = inputs.size()
        P = F.softmax(inputs, dim=1)        # [B, C, H, W]
        logP = F.log_softmax(inputs, dim=1) # [B, C, H, W]
        PlogP = P * logP                    # [B, C, H, W]

        if self.reduce_dim:                 # for loss computation -> ent_loss
            ent = -1.0 * PlogP.sum(dim=1)   # [B, 1, H, W]
            ent = ent / np.log(C)
        else:                               # for trainig discriminator -> Prob2Ix
            ent = -1.0 * PlogP              # [B, C, H, W]
            ent = ent / np.log(C)   
        
        # robust entropy calculation
        if self.charbonnier:     
            # compute robust entropy
            ent = ent ** 2.0 + 1e-8
            ent = ent ** self.ita # [B, 1, H, W] -or- [B, C, H, W]

        if self.reduce_dim: 
            ent = ent.mean()

        return ent

class LovaszSoftmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore=None):
        super(LovaszSoftmax, self).__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, inputs, targets):
        probas = F.softmax(inputs, dim=1) # B*C*H*W -> from logits to probabilities
        return lovasz_softmax(probas, targets, self.classes, self.per_image, self.ignore)

def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

def xloss(logits, labels, ignore=None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)


# --------------------------- HELPER FUNCTIONS ---------------------------
def isnan(x):
    return x != x
    
    
def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard