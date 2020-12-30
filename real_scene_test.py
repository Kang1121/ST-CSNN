import time

import torch
from PIL import Image, ImageOps
from preprocessing import preprocess
from torch.utils.data import Dataset
import numpy as np
from functions import network
import torch.nn.functional as F
from torchvision import transforms


def get_pos():
    """
        pos, box dimension. D1 = frame length; D2 = vehicle number in one frame; D3 = position.
        frame is a List of length(frame sequence), element in it means vehicle number in one frame.
    """
    pos, box, frame = preprocess()
    # for i in range(1, frame):
    return pos, box, frame


def get_img(path, name, st=None, length=None):
    """

    :return:
    """
    '''
        Use two dicts to record start frame and frame length
        '''

    datas = {}

    pos, box, frame = preprocess(name=name)
    # remove redundant box
    for i in range(len(box)):
        if box[i] is not None:
            for j in range(len(box[i])):
                for k in range(j + 1, len(box[i])):
                    if (abs(box[i][j][0] - box[i][k][0]) < 10) \
                        & (abs(box[i][j][1] - box[i][k][1]) < 10) \
                        & (abs(box[i][j][2] - box[i][k][2]) < 10) \
                            & (abs(box[i][j][3] - box[i][k][3]) < 10):
                        box[i][k][0], box[i][k][1], box[i][k][2], box[i][k][3] = -1, -1, -1, -1
                        pos[i][k][0], pos[i][k][1], pos[i][k][2] = -1, -1, -1

    # for i in range(len(box)):
    #     print(i, (box[i]))
    print(len(box))

    for i in range(length[name]):
        datas[i] = []
        filepath = path + '/' + str(i + st[name]) + '.jpg'
        s = Image.open(filepath).convert('RGB')
        if box[i] is not None:
            for j in range(len(box[i])):
                # print(box[i][j][0], box[i][j][1], box[i][j][2], box[i][j][3])
                # if box[i][j][0] != -1:
                    # s.show()
                img = s.crop((box[i][j][0], box[i][j][1], box[i][j][2], box[i][j][3]))
                img = img.resize((224, 224), Image.ANTIALIAS)
                datas[i].append(img)
                # if i > 1700:
                #     img.show()
                # else:
                #     datas[i].append(None)
        else:
            datas[i].append(None)

    return datas, pos


def test_set(net, name, threshold=0.5, st=None, length=None):

    data_transforms = transforms.Compose([
        transforms.RandomAffine(15),
        transforms.ToTensor()
    ])

    data, pos = get_img('E:/frames/' + name.split('_')[0] + '/', name=name, st=st, length=length)

    # construct dataset; compare frame by frame
    # d_r1r2 < r1 + r2
    # count is the result.
    count = 0
    # print(len(data))
    # exit(0)
    for i in range(len(data) - 1):

        if data[i + 1][0] is None:
            # print(i + 1, "no f2")
            continue
        elif data[i][0] is None:
            ### count all in frame l2
            # print(i + 1, "no f1")
            for j in range(len(pos[i + 1])):
                if pos[i + 1][j][0] != -1:
                    count += 1
            continue

        l2 = len(data[i + 1])
        l1 = len(data[i])

        pair = []
        tmp = []

        ### vehicles in frame 2
        no_f2 = True
        for j in range(l2):
            if pos[i + 1][j][0] != -1:
                no_f2 = False
                no_near = True
                no_same = True
                for k in range(l1):
                    if pos[i][k][0] != -1:
                        dis = np.sqrt((pos[i+1][j][1] - pos[i][k][1]) ** 2 + (pos[i+1][j][0] - pos[i][k][0]) ** 2)
                        r1r2 = pos[i+1][j][2] + pos[i][k][2]
                        if dis < r1r2 * 1.2:
                            no_near = False
                            tmp.append(data_transforms(data[i + 1][j]))
                            tmp.append(data_transforms(data[i][k]))
                            pair.append(tmp)
                            tmp = []
                if no_near:
                    count += 1
                    # print(i + 1, "no near")
                else:
                    # use SNN to compare
                    for ii in range(len(pair)):
                        pair[ii][0], pair[ii][1] = torch.unsqueeze(pair[ii][0], dim=0).float(), torch.unsqueeze(pair[ii][1], dim=0).float()
                        out, _, _ = net.forward(pair[ii][0].cuda(), pair[ii][1].cuda())
                        out = torch.sigmoid(out)
                        # if i > 650:
                        #     print(i + 1, out.item())
                        # if i > 670:
                        #     exit(0)
                        if out > threshold:
                            no_same = False
                    if no_same:
                        count += 1
                    #     print(i + 1, "no same")
                    # else:
                    #     print(i + 1, "same")
                pair = []
        # if no_f2:
        #     print(i + 1, "all -1 in f2")
        # print(i + 1, count)
        # print(i, len(pair), len(pos[i]))
    return count


net = network('squeezenet1_1').cuda()
net.load_state_dict(torch.load('E:/超算/backup_V0/saved_models/squeezenet1_1/squeezenet1_1-epoch-106.pt', map_location='cuda:0'))

st = {'0b_1': 1, '0b_2': 1862, '0b_3': 3721, '0a_1': 1, '0a_2': 1874, '0a_3': 3747, '0a_4': 5620, '45a_1': 1,
      '45a_2': 1862, '45a_3': 3599, '45a_4': 5401,  '45b_1': 1}

length = {'0b_1': 1860, '0b_2': 1858, '0b_3': 1856, '0a_1': 1873, '0a_2': 1872, '0a_3': 1873, '0a_4': 1871,
          '45a_1': 1864, '45a_2': 1739, '45a_3': 1804, '45a_4': 1802, '45b_1': 2125}

# input video title
for i in ['0b_1', '45a_1', '45a_2', '45a_3', '45a_4']:
    start = time.time()
    count = test_set(net, name=i, threshold=0.4, st=st, length=length)
    print(i, count)
    print(time.time() - start)


# for i in range(len(pair)):
#     print(i)
#     pair[i][0].show()
#     pair[i][1].show()
# class Test(Dataset):
#
#     def __init__(self, dataPath, transform=None, times=100, way=166):
#         np.random.seed(1)
#         super(Test, self).__init__()
#         self.transform = transform
#         self.times = times
#         self.way = way
#         self.img1 = None
#         self.c1 = None
#
#
#     def __len__(self):
#         return self.times * self.way
#
#     def __getitem__(self, index):
#         idx = index % self.way
#         label = None
#
#
#         if self.transform:
#             # print(type(self.img1))
#             # print(type(img2))
#             img1 = self.transform(self.img1)
#             img2 = self.transform(self.img2)
#         # print('datas.shape = ', len(self.datas))
#         # print('img1 = ', img1.shape)
#         # print('img2 = ', img2.shape)
#         return img1, img2



