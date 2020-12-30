from main import *
import torch
import matplotlib.pyplot as plt
# from train import net
net = Siamese()
net.load_state_dict(torch.load('/home/yk/siamese-pytorch/models/model-inter-96001.pt'))
net.cuda()

from collections import OrderedDict
# model.load_state_dict(net)
testset = OmniglotTest('/media/yk/Academy/image/evaluation', transform=transforms.ToTensor(), times=10,
                             way=200)
testLoader = DataLoader(testset, batch_size=200, shuffle=False, num_workers=0)

right = 0
error = 0
count = 0


for _, (test1, test2) in enumerate(testLoader, 1):
    count = count + 1
    # print('test1 = ', test1.shape)
    # print('test2 = ', test2.shape)
    # print(test1.shape)
    test1, test2 = Variable(test1.cuda()), Variable(test2.cuda())
    output = net.forward(test1, test2).data.cpu().numpy()
    # print(test1.shape)
    for i in range(len(output)):
        if abs(output[i] - (i + 1) % 2) < 0.5:
            right = right + 1
        else:
            error = error + 1

    # Visualization
    if count == 1:
        img1_show = test1[:10, :, :, :]
        img2_show = test2[:10, :, :, :]

        # for i in range(10):
        #     plt.figure()
        #     print(np.round(output[i]))
        #     # plt.subplot(2, 2, 2 * i + 1)
        #
        #     plt.imshow(img1_show[i].cpu().numpy())
        #     plt.xticks([])
        #     plt.yticks([])
        #     plt.figure()
        #     # plt.subplot(2, 2, 2 * i + 2)
        #     plt.imshow(img2_show[i].cpu().numpy())
        #     plt.xticks([])
        #     plt.yticks([])
        # plt.show()
        i = 0
        print(img1_show[i].shape)
        a = img1_show[i].transpose((1, 2, 0))
        print(a.shape)
# print('i = ', i)
# print('output = ', output)
# print('pred = ', pred)
print('*' * 70)
print('[%d]\tTest set\tcorrect:\t%d\terror:\t%d\tprecision:\t%f' % (
    1, right, error, right * 1.0 / (right + error)))
print('*' * 70)
