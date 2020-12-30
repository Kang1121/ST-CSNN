import time
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from eval import *
import torch
from args import opt
from torch.optim import lr_scheduler
from models import *
from models.mysiamese import Siamese


#def train(trainLoader, net):
def train(net):
    loss_soft = torch.nn.CrossEntropyLoss()
    loss_snn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, weight_decay=opt.set_weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    optimizer.zero_grad()

    for epoch in range(opt.epoch, 201):

        f = open('E:/1VERIWILD/4use/loss_record/' + opt.arch + '.txt', 'a+')
        print('*' * 70)
        print('Epoch: ', epoch)
        print('lr:%s' % optimizer.state_dict().get('param_groups')[0].get('lr'))
        loss_val = 0
        time_start = time.time()

        for batch_id, (img1, img2, label, label_id1, label_id2) in enumerate(trainLoader, 1):

            if opt.cuda:
                img1, img2, label, label_id1, label_id2 \
                    = Variable(img1.cuda()), Variable(img2.cuda()), Variable(label.cuda()), Variable(
                    label_id1.cuda()), Variable(label_id2.cuda())
            else:
                img1, img2, label = Variable(img1), Variable(img2), Variable(label)
            # out 1-dimensional probability
            # feature 40671-dimensional soft-max
            out, feature1, feature2 = net.forward(img1, img2)
            loss = opt.alpha * loss_snn(out, label) + \
                   (1 - opt.alpha) * (loss_soft(feature1, label_id1) + loss_soft(feature2, label_id2))
            loss_val += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(label_id2.size())
            if batch_id % opt.show_every == 0:
                print(opt.alpha * loss_snn(out, label))
                print((1 - opt.alpha) * (loss_soft(feature1, label_id1) + loss_soft(feature2, label_id2)))
                print('[%d]\tloss:\t%.5f\ttime lapsed:\t%.2f s' % (
                    batch_id, loss_val / opt.show_every, time.time() - time_start))
                f.write(str(epoch) + '/' + str(loss_val / opt.show_every) + '\n')
                loss_val = 0
                time_start = time.time()

        if epoch % opt.save_every == 0:  # and epoch != 0:
            # validation(queryLoader, galleryLoader, net)
            torch.save(net.state_dict(), opt.model_path + '/' + opt.arch + '/' + opt.arch + '-epoch-' + str(epoch) + '-' + str(opt.alpha) + ".pt")
            # torch.save(net.state_dict(),
            #             opt.model_path + '/' + opt.arch + '/' + opt.arch + '-epoch-' + str(epoch) + ".pt")

        f.close()
        break
        # scheduler.step()


def validation(queryset, galleryset, net):

    (g_feature, g_label), (q_feature, q_label) = concatenate(galleryset, net), concatenate(queryset, net)

    t = np.zeros(10)
    for i in range(q_feature.shape[0]):
        label = torch.tensor([i]).cuda()

        dis = -(F.pairwise_distance(q_feature[i].expand_as(g_feature), g_feature))
        dis = dis.resize_(1, dis.shape[0])

        topkk = []
        for j in range(1, 11):
            topkk.append(j)
        prec = accuracy(i, dis.data, g_label.data, topk=topkk)
        # print(dis)
        # top1.update(prec1.item(), label.size(0))
        # top5.update(prec5.item(), label.size(0))
        # top30.update(prec30.item(), label.size(0))

        for j in range(10):
            t[j] += prec[j]

    for i in range(10):
        t[i] /= 3000

    print('Validation\ttop1:%.2f\ttop5:%.2f\ttop30:%.2f' % (t[0], t[4], t[9]))
    # print(t)
    f = open('C:/Users/yk/Desktop/alpha.txt', 'a+')
    f.write(str(t) + '\n')
    f.close()


def concatenate(dataset, net):
    total = None
    lab = None
    for batch_id, (img, label) in enumerate(dataset, 1):
        # print(batch_id)
        with torch.no_grad():
            img, label = Variable(img.cuda()), Variable(label.cuda())
            feature = net.forward(img, None)
        if total is None:
            # print(feature)
            total = feature
            lab = label
        else:
            total = torch.cat([total, feature], dim=0)
            lab = torch.cat([lab, label], dim=0)
        # if batch_id == 11:
        #     total = total.cpu().numpy()
        #     lab = lab.cpu().numpy()
        #     np.savetxt("C:/Users/yk/Desktop/feature.txt", total)
        #     np.savetxt("C:/Users/yk/Desktop/lab.txt", lab)
        #     exit(0)
    return total, lab


def network(arch=None):
    if arch is not None:
        opt.arch = arch
    if opt.arch == 'alexnet':
        net = alexnet()
    elif opt.arch == 'squeezenet1_0':
        net = squeezenet1_0()
    elif opt.arch == 'squeezenet1_1':
        net = squeezenet1_1()
    elif opt.arch == 'densenet121':
        net = densenet121(pretrained=False)
    elif opt.arch == 'densenet169':
        net = densenet169()
    elif opt.arch == 'densenet201':
        net = densenet201()
    elif opt.arch == 'densenet161':
        net = densenet161()
    elif opt.arch == 'vgg11':
        net = vgg11()
    elif opt.arch == 'vgg13':
        net = vgg13()
    elif opt.arch == 'vgg16':
        net = vgg16()
    elif opt.arch == 'vgg19':
        net = vgg19()
    elif opt.arch == 'vgg11_bn':
        net = vgg11_bn()
    elif opt.arch == 'vgg13_bn':
        net = vgg13_bn()
    elif opt.arch == 'vgg16_bn':
        net = vgg16_bn()
    elif opt.arch == 'vgg19_bn':
        net = vgg19_bn()
    elif opt.arch == 'resnet18':
        net = resnet18()
    elif opt.arch == 'resnet34':
        net = resnet34()
    elif opt.arch == 'resnet50':
        net = resnet50()
    elif opt.arch == 'resnet101':
        net = resnet101()
    elif opt.arch == 'resnet152':
        net = resnet152()
    else:
        net = Siamese()

    return net


# def test(queryloader, galleryloader, net):
#     validation(queryloader, galleryloader, net)