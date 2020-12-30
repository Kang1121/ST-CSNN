import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--cuda", type=bool, default=True, help="use cuda")
parser.add_argument("--train_path", type=str, default="E:/1VERIWILD/4use/train", help="training folder")
parser.add_argument("--test_path", type=str, default="E:/DeepLearning/VERI-Wild/images", help='path of testing folder')
parser.add_argument("--way", type=int, default=100, help="how much way one-shot learning")
parser.add_argument("--times", type=int, default=10, help="number of samples to test accuracy")
parser.add_argument("--workers", type=int, default=0, help="number of dataLoader workers")
parser.add_argument("--batch_size", type=int, default=20, help="number of batch size")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--show_every", type=int, default=180, help="show result after each show_every iter.")
parser.add_argument("--save_every", type=int, default=2, help="save model after each save_every iter.")
parser.add_argument("--test_every", type=int, default=100, help="test model after each test_every iter.")
parser.add_argument("--model_path", type=str, default="E:/超算/backup_V0/saved_models", help="path to store model")
parser.add_argument("--set_weight_decay", type=float, default=0, help="weight_decay each iteration")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--arch", type=str, default='Siamese', help="model choice")
parser.add_argument("--alpha", type=float, default=0.9, help="coefficient of loss function")
parser.add_argument("--train", type=bool, default=False, help="choose to train")
parser.add_argument("--test", type=bool, default=False, help="choose to test")
parser.add_argument("--load", type=str, default=None, help="load pre-trained model")
parser.add_argument("--epoch", type=int, default=0, help="choose epoch to start")
parser.add_argument("--simple", type=bool, default=False, help="choose whether use small dataset")
opt = parser.parse_args()

# net = Siamese()
# print(net)
# a = 1.0
# print(type(a))
# print(str(a))
import numpy
# t = numpy.zeros(10)
# f = open('C:/Users/yk/Desktop/alpha.txt', 'a+')
# f.write(str(t) + '\n')
# f.close()
# import torch
# a = torch.ones((2, 5))
# s = sum(a[1])
# b = sum(a[:])
# assert s == 5.0
# print(s)
# print(b)
# for i in range(800, 2400):
#     print(i / 2400)
