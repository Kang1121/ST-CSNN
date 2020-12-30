import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from args import opt

# you need to download the models to ~/.torch/models
# model_urls = {
#     'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
# }
models_dir = os.path.expanduser('~/.torch/models')
model_name = 'alexnet-owt-4df8aa71.pth'


class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta


    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x


class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()
        self.num_classes = 10000
        if opt.simple:
            self.num_classes = 1000

        def discriminator_block(in_filters, out_filters, kernel_size, stride, padding, bn=False, lrn=True, pool=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, kernel_size, stride, padding), nn.LeakyReLU(0.2, inplace=True)]

            if pool:
                block.append(nn.MaxPool2d(kernel_size=3, stride=2))
            if lrn:
                block.append(nn.LocalResponseNorm(2))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(3, 96, 11, 4, 2),
            *discriminator_block(96, 256, 5, 1, 2),
            *discriminator_block(256, 384, 3, 1, 1, lrn=False, pool=False),
            *discriminator_block(384, 384, 3, 1, 1, lrn=False, pool=False),
            *discriminator_block(384, 256, 3, 1, 1),

        )

        self.linear_blocks = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4096, self.num_classes),
        )
        '''
        self.features = nn.Sequential(
            # input shape is 224 x 224 x 3
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),  # shape is 55 x 55 x 64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # shape is 27 x 27 x 64
            LRN(local_size=5, alpha=0.0001, beta=0.75),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),  # shape is 27 x 27 x 192
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # shape is 13 x 13 x 192
            LRN(local_size=5, alpha=0.0001, beta=0.75),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),  # shape is 13 x 13 x 384
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),  # shape is 13 x 13 x 256
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # shape is 13 x 13 x 256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)  # shape is 6 x 6 x 256
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        '''
        self.out = nn.Linear(self.num_classes, 1)
        # self.soft = nn.Linear(1000, 1000)

    # def forward_one(self, x):
    #     x = self.features(x)
    #     x = x.view(x.size(0), 256 * 6 * 6)
    #     x = self.classifier(x)
    #     return x

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
        # feature1 = self.soft(x1)
        # feature2 = self.soft(x2)
            return out, x1, x2  # feature1, feature2
        else:
            return x1


def alexnet(pretrained=False, **kwargs):
    """
    AlexNet model architecture 

    Args:
        pretrained (bool): if True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name)))
    return model