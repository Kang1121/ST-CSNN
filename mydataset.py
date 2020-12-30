import torch
from torch.utils.data import Dataset, DataLoader
import os
from numpy.random import choice as npc
import numpy as np
import random
from PIL import Image
from shutil import move
import linecache
from args import opt


class OmniglotTrain(Dataset):

    def __init__(self, dataPath, transform=None):
        super(OmniglotTrain, self).__init__()
        np.random.seed(0)

        self.transform = transform
        self.datas, self.num_classes = self.loadToMem(dataPath)

        if opt.simple:
            self.len = 9524
            file = open("E:/1VERIWILD/4use/simpletrain.txt")
        else:
            self.len = 93085
            file = open("E:/1VERIWILD/4use/train.txt")

        self.lines = []
        for line in file:
            self.lines.append(line)
        file.close()

    def loadToMem(self, dataPath):
        print("begin loading training dataset to memory")
        datas = {}
        agrees = [0]
        idx = 1
        for agree in agrees:
            for alphaPath in os.listdir(dataPath):
                # for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
                datas[idx] = []
                # temp = 0
                for samplePath in os.listdir(os.path.join(dataPath, alphaPath)):
                    filePath = os.path.join(dataPath, alphaPath, samplePath)
                    s = Image.open(filePath).rotate(agree).convert('RGB')
                    s = s.resize((224, 224), Image.ANTIALIAS)
                    datas[idx].append(s)
                #     temp += 1
                # datas[idx] = temp
                idx += 1
        print("finish loading training dataset to memory")

        return datas, idx

    def __len__(self):

        return self.len

    def __getitem__(self, index):

        # index starts from 0
        # get image from same class
        temp = self.lines[index]
        id1 = int(temp.split('/')[0])
        img = int(temp.split('/')[1])
        image1 = self.datas[id1][img]
        if index % 2 == 0:
            label = 1.0
            id2 = id1
            image2 = random.choice(self.datas[id1])
        # get image from different class
        else:
            label = 0.0
            id2 = random.randint(1, self.num_classes - 1)
            while id1 == id2:
                id2 = random.randint(1, self.num_classes - 1)
            image2 = random.choice(self.datas[id2])

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, torch.from_numpy(np.array([label], dtype=np.float32)), \
               torch.from_numpy(np.array(id1, dtype=np.int64)), torch.from_numpy(np.array(id2, dtype=np.int64))


class Gallery(Dataset):

    def __init__(self, dataPath, transform=None):
        super(Gallery, self).__init__()
        np.random.seed(0)
        # self.dataset = dataset
        self.transform = transform
        self.datas, self.num_classes = self.loadToMem(dataPath)
        self.x, self.y = 0, 0

    def loadToMem(self, dataPath):
        print("begin loading Galleryset to memory")
        datas = {}
        idx = 0
        datas[idx] = []
        file = open("E:/1VERIWILD/train_test_split/test_10000.txt")
        # file = open("C:/Users/yk/Desktop/generate.txt")
        while 1:
            lines = file.readlines(100000)
            if not lines:
                break
            flag = 1
            last_id = 0
            for line in lines:
                # for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
                # for idx in range(1, 3001):
                id = line.split('/')[0]

                # print(id, last_id)
                if id == last_id or flag == 1:
                    flag = 0
                else:
                    idx += 1
                    datas[idx] = []

                filepath = dataPath + '/' + line.strip('\n') + '.jpg'
                s = Image.open(filepath).convert('RGB')
                s = s.resize((224, 224), Image.ANTIALIAS)
                datas[idx].append(s)
                last_id = id

        file.close()
        print("finish loading Galleryset to memory")

        return datas, idx

    def __len__(self):
        return 38861

    def __getitem__(self, index):
        x, y = self.x, self.y
        l = len(self.datas[x])

        # index starts from 0
        # get image from same class

        image1 = self.datas[x][y]

        if self.transform:
            image1 = self.transform(image1)

        if y == l - 1:
            x += 1
            y = 0
        else:
            y += 1
        self.x, self.y = x, y
        if self.x == 3000:
            self.x, self.y = 0, 0
        return image1, torch.from_numpy(np.array(x, dtype=np.int64))


class Query(Dataset):

    def __init__(self, dataPath, transform=None):
        super(Query, self).__init__()
        np.random.seed(0)
        # self.dataset = dataset
        self.transform = transform
        self.datas, self.num_classes = self.loadToMem(dataPath)
        self.x, self.y = 0, 0

    def loadToMem(self, dataPath):
        print("begin loading Queryset to memory")
        datas = {}
        idx = 0
        datas[idx] = []
        file = open("E:/1VERIWILD/train_test_split/test_3000_query.txt")

        while 1:
            lines = file.readlines(100000)
            if not lines:
                break
            flag = 1
            last_id = 0
            for line in lines:
                # for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
                # for idx in range(1, 3001):
                id = line.split('/')[0]

                # print(id, last_id)
                if id == last_id or flag == 1:
                    flag = 0
                else:
                    idx += 1
                    datas[idx] = []

                filepath = dataPath + '/' + line.strip('\n') + '.jpg'
                s = Image.open(filepath).convert('RGB')
                s = s.resize((224, 224), Image.ANTIALIAS)
                datas[idx].append(s)
                last_id = id

        file.close()
        print("finish loading Queryset to memory")

        return datas, idx

    def __len__(self):
        return 3000

    def __getitem__(self, index):
        x, y = self.x, self.y
        l = len(self.datas[x])

        # index starts from 0
        # get image from same class

        image1 = self.datas[x][y]

        if self.transform:
            image1 = self.transform(image1)

        if y == l - 1:
            x += 1
            y = 0
        else:
            y += 1
        self.x, self.y = x, y
        if self.x == 3000:
            self.x, self.y = 0, 0
        return image1, torch.from_numpy(np.array(x, dtype=np.int64))


class OmniglotTest(Dataset):

    def __init__(self, dataPath, transform=None, times=100, way=166):
        np.random.seed(1)
        super(OmniglotTest, self).__init__()
        self.transform = transform
        self.times = times
        self.way = way
        self.img1 = None
        self.c1 = None
        self.datas, self.num_classes = self.loadToMem(dataPath)

    def loadToMem(self, dataPath):
        print("begin loading test dataset to memory")
        datas = {}
        idx = 0
        for alphaPath in os.listdir(dataPath):
            # for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
                datas[idx] = []
                for samplePath in os.listdir(os.path.join(dataPath, alphaPath)):
                    filePath = os.path.join(dataPath, alphaPath, samplePath)
                    s = Image.open(filePath).convert('RGB')
                    s = s.resize((224, 224), Image.ANTIALIAS)
                    datas[idx].append(s)
                idx += 1
        print("finish loading test dataset to memory")
        return datas, idx

    def __len__(self):
        return self.times * self.way

    def __getitem__(self, index):
        idx = index % self.way
        label = None

        # generate image pair from same class
        self.c1 = random.randint(0, self.num_classes - 1)
        self.img1 = random.choice(self.datas[self.c1])

        if idx % 2 == 0:
            img2 = random.choice(self.datas[self.c1])
        # generate image pair from different class
        else:
            c2 = random.randint(0, self.num_classes - 1)
            while self.c1 == c2:
                c2 = random.randint(0, self.num_classes - 1)
            img2 = random.choice(self.datas[c2])

        if self.transform:
            # print(type(self.img1))
            # print(type(img2))
            img1 = self.transform(self.img1)
            img2 = self.transform(img2)
        # print('datas.shape = ', len(self.datas))
        # print('img1 = ', img1.shape)
        # print('img2 = ', img2.shape)
        return img1, img2


class Omniglotreal_test(Dataset):

    def __init__(self, dataPath, transform=None, times=100, way=166):
        np.random.seed(1)
        super(Omniglotreal_test, self).__init__()
        self.transform = transform
        self.times = times
        self.way = way
        self.img1 = None
        self.c1 = None
        self.datas, self.num_classes = self.loadToMem(dataPath)

    def loadToMem(self, dataPath):
        print("begin loading test dataset to memory")
        datas = {}
        idx = 0
        datas = []
        for alphaPath in os.listdir(dataPath):
            # for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
                # for samplePath in os.listdir(os.path.join(dataPath, alphaPath)):
                filePath = os.path.join(dataPath, alphaPath)
                print(filePath)
                s = Image.open(filePath).convert('RGB')
                s = s.resize((105, 105), Image.ANTIALIAS)
                datas.append(s)
                # print(len(datas))
        print("finish loading test dataset to memory")
        return datas, idx

    def __len__(self):
        return self.times * self.way

    def __getitem__(self, index):
        idx = index % self.way
        label = None

        # # generate image pair from same class
        # self.c1 = random.randint(0, self.num_classes - 1)
        # self.img1 = random.choice(self.datas[self.c1])
        #
        # if idx % 2 == 0:
        #     img2 = random.choice(self.datas[self.c1])
        # # generate image pair from different class
        # else:
        #     c2 = random.randint(0, self.num_classes - 1)
        #     while self.c1 == c2:
        #         c2 = random.randint(0, self.num_classes - 1)
        #     img2 = random.choice(self.datas[c2])
        # print(idx)
        # print(len(self.datas))
        self.img1 = self.datas[2 * idx]
        img2 = self.datas[2 * idx + 1]
        # print(img2.shape)

        if self.transform:
            # print(type(self.img1))
            # print(type(img2))
            img1 = self.transform(self.img1)
            img2 = self.transform(img2)
        # print('datas.shape = ', len(self.datas))
        # print('img1 = ', img1.shape)
        # print('img2 = ', img2.shape)
        return img1, img2


def mkdir(path):

    for i in range(1, 10001): #创建文件个数
        file_name = path + str(i).zfill(5)
        os.mkdir(file_name)


def train_test_set_split(path):
    for id in os.listdir(path):
        idpath = os.path.join(path, id)
        for sample in os.listdir(idpath):
            samplepath = os.path.join(idpath, sample)

            src_path = samplepath
            dst_path = 'E:/1VERIWILD/4use/test/' + str(id) + '/'
            move(src_path, dst_path)
            break


def class_id_txt(path):


    f = open('E:/1VERIWILD/4use/test.txt', 'w')
    for id in os.listdir(path):
        count = 0
        idpath = os.path.join(path, id)
        for sample in os.listdir(idpath):

            # samplepath = os.path.join(idpath, sample)
            f.write(str(id) + '/' + str(count).split('.')[0] + '\n')
            count += 1
    f.close()


if __name__ == '__main__':

    # mkdir('E:/1VERIWILD/4use/test/')
    # train_test_set_split('E:/1VERIWILD/4use/train/')
    class_id_txt('E:/1VERIWILD/4use/test/')
