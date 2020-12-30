from torchvision import transforms
from mydataset import OmniglotTrain, Gallery, Query
from torch.utils.data import DataLoader
import os
from functions import *
from args import opt


if __name__ == '__main__':

    data_transforms = transforms.Compose([
        transforms.RandomAffine(15),
        transforms.ToTensor()
    ])



    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device_ids = [0, 1, 2, 3]

    if opt.train:

        if opt.simple:
            opt.train_path = 'E:/1VERIWILD/4use/simple_train'
            opt.save_every = 20

        net = network()
        if opt.load:
            net.load_state_dict(torch.load('E:/超算/backup_V0/saved_models/' + opt.arch + '/' + opt.arch + '-epoch-' + opt.load + '.pt', map_location='cuda:0'))
            opt.epoch = int(opt.load) + 1

        print(net)
        print(opt.alpha)

        if opt.cuda:
            net.cuda()
        net.train()

        # trainSet = OmniglotTrain(opt.train_path, transform=data_transforms)
        # trainLoader = DataLoader(trainSet, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)
        #
        # train(trainLoader, net)
        train(net)

    if opt.test:

        gallerySet = Gallery(opt.test_path, transform=transforms.ToTensor())
        galleryLoader = DataLoader(gallerySet, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)
        querySet = Query(opt.test_path, transform=transforms.ToTensor())
        queryLoader = DataLoader(querySet, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)

        # net = network()
        # for i in range(4):
        #     net.load_state_dict(torch.load('E:/超算/backup_V0/saved_models/' + opt.arch + '/' + opt.arch + '-epoch-' + str(180 + 20 * i) + '.pt',
        #                                         map_location='cuda:0'))
        #     validation(queryLoader, galleryLoader, net.cuda())
        ll = ['alexnet/alexnet-epoch-0-0.9', 'squeezenet1_1/squeezenet1_1-epoch-0-0.9', 'Siamese/Siamese-epoch-0-0.9']
        netw = ['alexnet', 'squeezenet1_1', 'Siamese']
        for i in range(3):
            opt.arch = netw[i]
            net = network()
            net.load_state_dict(torch.load(
                'E:/超算/backup_V0/saved_models/' + ll[i] + '.pt',
                map_location='cuda:0'))
            time_start = time.time()
            validation(queryLoader, galleryLoader, net.cuda())
            print('time', time.time() - time_start)

        # print('Model? Input . to stop, otherwise input model name')
        # getnet = input()
        # while getnet != '.':
        #     print('Simple? y or n')
        #     get = input()
        #     if get == 'y':
        #         opt.simple = True
        #     else:
        #         opt.simple = False
        #     net = network(getnet)
        #     print('Epoch?')
        #     get = input()
        #     net.load_state_dict(torch.load('E:/超算/backup_V0/saved_models/' + opt.arch + '/' + opt.arch + '-epoch-' +
        #                                    get + '.pt', map_location='cuda:0'))
        #     validation(queryLoader, galleryLoader, net.cuda())
        #     print('Model? Input . to stop, otherwise input model name')
        #     getnet = input()


        #     if loss.item() < best_loss:
        #         best_loss = loss.item()
        #     if loss.item() > 4 * best_loss or optimizer.state_dict().get('param_groups')[0].get('lr') > 1e-3:
        #         epoch = 14
        #         break
        #
        #     lr.append(optimizer.state_dict().get('param_groups')[0].get('lr'))
        #     losses.append(loss.item())
        #     scheduler.step()
        #     print(loss.item())
        #     print(optimizer.state_dict().get('param_groups')[0].get('lr'))
        #
        # if epoch == 14:
        #     plt.figure()
        #     plt.xticks(np.log([1e-6, 1e-5, 1e-4, 1e-3]), (1e-6, 1e-5, 1e-4, 1e-3))
        #     plt.xlabel('learning rate')
        #     plt.ylabel('loss')
        #     plt.plot(np.log(lr), losses)
        #     plt.show()
        #     plt.figure()
        #     plt.xlabel('num iterations')
        #     plt.ylabel('learning rate')
        #     plt.plot(lr)
        #
        #     exit(0)

    '''
        if batch_id % opt.test_every == 0:
            right, error = 0, 0
            for _, (validation1, validation2) in enumerate(validationLoader, 1):

                if opt.cuda:
                    validation1, validation2 = validation1.cuda(), validation2.cuda()
                validation1, validation2 = Variable(validation1), Variable(validation2)
    
                out = net.forward(validation1, validation2)
                out = torch.sigmoid(out)

                # print(output.shape)
                for i in range(len(out)):
                    if abs(out[i] - (i + 1) % 2) < 0.5:
                        right = right + 1
                    else:
                        error = error + 1


            # print('i = ', i)
            # print('output = ', output)
            # print('pred = ', pred)
            print('*'*70)
            print('[%d]\tValidation set\tcorrect:\t%d\terror:\t%d\tprecision:\t%f'%(batch_id, right, error, right*1.0/(right+error)))
            print('*'*70)
            print('Learning Rate: ', opt.lr)
            print('*' * 70)
    '''
        # if batch_id % 800 == 0:
        #     queue.append(right * 1.0 / (right + error))
        #     l1 = len(queue)
        #     count = 0
        #     if l1 >= 20:
        #         for j1 in range(19):
        #             if queue[l1 - 20 + j1] >= queue[l1 - 20 + j1 + 1]:
        #                 count = count + 1
        #     if count == 20:
        #         # torch.save(net.state_dict(), opt.model_path + '/model-inter-' + str(batch_id + 1) + ".pt")
        #         break


        # train_loss.append(loss_val)
    #  learning_rate = learning_rate * 0.95
    
    # with open('train_loss', 'wb') as f:
    #     pickle.dump(train_loss, f)
    #
    # acc = 0.0
    # for d in queue:
    #     acc += d
    # print("#"*70)
    # print("final accuracy: ", acc/20)
