import torch
import torch.nn as nn
import torch.nn.functional as F
from args import opt


class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.num_classes = 10000
        if opt.simple:
            self.num_classes = 1000

        def discriminator_block(in_filters, out_filters, kernel_size, stride, padding, bn=False, lrn=True, pool=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, kernel_size, stride, padding), nn.LeakyReLU(0.2, inplace=True)]

            if pool:
                block.append(nn.MaxPool2d(2))
            if lrn:
                block.append(nn.LocalResponseNorm(2))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(3, 96, 7, 2, 0),
            *discriminator_block(96, 256, 5, 2, 1),
            *discriminator_block(256, 512, 3, 1, 1, lrn=False, pool=False),
            *discriminator_block(512, 512, 3, 1, 1, lrn=False, pool=False),
            *discriminator_block(512, 512, 3, 1, 1),
        )

        self.linear_blocks = nn.Sequential(
            nn.Dropout2d(0.25),
            nn.Linear(18432, 4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4096, self.num_classes),
        )

        self.out = nn.Linear(self.num_classes, 1)
        # self.soft = nn.Softmax(dim=1)

    def forward_one(self, x):
        x = self.conv_blocks(x)
        x = x.view(x.size()[0], -1)
        x = self.linear_blocks(x)
        return x

    def forward(self, x1, x2=None):
        x1 = self.forward_one(x1)
        if x2 is not None:
            x2 = self.forward_one(x2)
            dis = torch.abs(x1 - x2)
            out = self.out(dis)
            # x1 = self.soft(x1)
            # x2 = self.soft(x2)
            return out, x1, x2  # feature1, feature2
        else:
            return x1


def loss_regularization(output, label):

    return loss


# for test
if __name__ == '__main__':
    arch = 'alex'
    f = open('E:/1VERIWILD/4use/loss_record/' + arch + '.txt', 'w')
    # net = Siamese()
    # print(net)
    # print(list(net.parameters()))
