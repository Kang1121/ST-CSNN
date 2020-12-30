from __future__ import print_function, absolute_import

__all__ = ['accuracy', 'AverageMeter']


def accuracy(i, output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)

    res = []
    # print(pred)
    for k in topk:
        flag = 0.0
        for j in range(k):
            if target[pred[0][j]] == i:
                flag = 1.0
                break
        res.append(flag)

    return res


DEBUG = False


def map(num_of_good, rows_good):
    for i in range(num_of_good):
        # The gap between every recall value
        d_recall = 1.0 / num_of_good
        precision = (i + 1)*1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            # The last precision, so i = i + 1 - 1
            old_precision = i * 1.0 / rows_good[i]
        else:
            # Avoid zero
            old_precision = 1.0
        # ap is the area under the p-r curve
        ap = ap + d_recall * (old_precision + precision) / 2
    return ap


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


import torch


a = torch.tensor([[99, 999, 9, 0, 90]])
l = torch.tensor([4, 1, 0, 2, 3])
#a = a.t()
#print(a.size())
accuracy(4, a, l, topk=(1, 3, 5))
