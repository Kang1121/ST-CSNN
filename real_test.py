from mydataset import Omniglotreal_test
from main import *
import torch
import matplotlib.pyplot as plt
import torch
# from train import net
net = Siamese()
net.load_state_dict(torch.load('/home/yk/siamese-pytorch/models/16000/1e-5/model-inter-11201.pt'))
net.cuda()

from collections import OrderedDict
# model.load_state_dict(net)
testset = Omniglotreal_test('/media/yk/Academy/image/realtest2', transform=transforms.ToTensor(), times=1,
                             way=2)
testLoader = DataLoader(testset, batch_size=2, shuffle=False, num_workers=0)

right = 0
error = 0
count = 0

for param in net.parameters():
    print(type(param.data), param.size())

for _, (test1, test2) in enumerate(testLoader, 1):
    count = count + 1
    # print('test1 = ', test1.shape)
    # print('test2 = ', test2.shape)
    # print(test1.shape)
    test1, test2 = Variable(test1.cuda()), Variable(test2.cuda())
    output = net.forward(test1, test2).data.cpu().numpy()
    img1_show = test1[:2, :, :, :]
    img2_show = test2[:2, :, :, :]
    for i in range(2):
        plt.figure()
        print(np.round(output[i]))
        print((output[i]))
        # plt.subplot(2, 2, 2 * i + 1)
        plt.imshow(img1_show[i][0].cpu().numpy())
        plt.xticks([])
        plt.yticks([])
        plt.figure()
        # plt.subplot(2, 2, 2 * i + 2)
        plt.imshow(img2_show[i][0].cpu().numpy())
        plt.xticks([])
        plt.yticks([])
    plt.show()
    # print(test1.shape)
    # for i in range(len(output)):
    #     if abs(output[i] - (i + 1) % 2) < 0.5:
    #         right = right + 1
    #         # print('r=', right)
    #     else:
    #         error = error + 1
    #         # print('e=', error)
    #
    # # Visualization
    # if count == 1:
    #     img1_show = test1[:20, :, :, :]
    #     img2_show = test2[:20, :, :, :]
    #
    #     for i in range(20):
    #         plt.figure()
    #         print(np.round(output[i]))
    #         print((output[i]))
    #         # plt.subplot(2, 2, 2 * i + 1)
    #         plt.imshow(img1_show[i][0].cpu().numpy())
    #         plt.xticks([])
    #         plt.yticks([])
    #         plt.figure()
    #         # plt.subplot(2, 2, 2 * i + 2)
    #         plt.imshow(img2_show[i][0].cpu().numpy())
    #         plt.xticks([])
    #         plt.yticks([])
    #     plt.show()




# print('i = ', i)
# print('output = ', output)
# print('pred = ', pred)
# print('*' * 70)
# print('[%d]\tTest set\tcorrect:\t%d\terror:\t%d\tprecision:\t%f' % (
#     1, right, error, right * 1.0 / (right + error)))
# print('*' * 70)
