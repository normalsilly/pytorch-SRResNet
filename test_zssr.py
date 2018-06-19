#import matlab.engine
import matlab
import argparse, os
import torch
import numpy as np
import time, math, glob
import scipy.io as sio
import cv2
import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from srresnet import _NetG
from dataset import DatasetFromHdf5
from torchvision import models
from random import shuffle
import torch.utils.model_zoo as model_zoo
from dataset import DataValSet_zssr, DataValSet_zssr_train
from torchvision import transforms
from os.path import join
from math import log10
import pandas as pd
import os

parser = argparse.ArgumentParser(description="PyTorch SRResNet Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model/model_srresnet.pth", type=str, help="model path") #TODO: change default model
parser.add_argument("--dataset", default="LR", type=str, help="dataset name, Default: LR")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--vgg_loss", action="store_true", help="Use content loss?")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--batchSize", type=int, default=1, help="testing batch size")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--step", type=int, default=200, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument("--nEpochs", type=int, default=200, help="number of epochs to train for")
parser.add_argument("--trainset", type=bool, default=True,  help="dataset name, true means use small image")


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr

def save_to_csv(filename, data):
    df = pd.DataFrame(data)
    df.to_csv(filename, header=False, index=False)

# TODO: update lr
def train(training_data_loader, optimizer, model, criterion, epoch):
    lr = adjust_learning_rate(optimizer, epoch-1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):

        input, target = Variable(batch[0]), Variable(batch[1],
                                                     requires_grad=False)

        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        output = model(input)
        loss = criterion(output, target)

        if opt.vgg_loss:
            content_input = netContent(output)
            content_target = netContent(target)
            content_target = content_target.detach()
            content_loss = criterion(content_input, content_target)

        optimizer.zero_grad()

        if opt.vgg_loss:
            netContent.zero_grad()
            content_loss.backward(retain_graph=True)

        loss.backward()

        optimizer.step()
    print("===> epoch[{}]: Loss: {:.5}".format(epoch, loss.data[0]))


def test(test_gen, model, criterion, SR_dir, log_file, is_origin_model, id):
    avg_psnr = 0

    for iteration, batch in enumerate(test_gen, 1):
        input,  target = Variable(batch[0], volatile=True), Variable(batch[1], volatile=True)
        input = input.cuda()
        target = target.cuda()

        Blur_SR = model(input)

        im_h = Blur_SR.cpu().data[0].numpy().astype(np.float32)
        im_h[im_h < 0] = 0
        im_h[im_h > 1.] = 1.

        SR = Variable((torch.from_numpy(im_h)).unsqueeze(0)).cuda()

        result = transforms.ToPILImage()(SR.cpu().data[0])
        if is_origin_model:
            path = join(SR_dir, '{}_origin.jpg'.format(id))
        else:
            path = join(SR_dir, '{}_new.jpg'.format(id))
        result.save(path)
        mse = criterion(SR, target)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
        print('iter:' + str(iteration))
        print('MSE:' + str(mse.data[0]))
        # if is_origin_model:
        #     save_to_csv(join(SR_dir, '{}_origin.csv'.format(id)), (SR-target)[0])
        # else:
        #     save_to_csv(join(SR_dir, '{}_new.csv'.format(id)), (SR-target)[0])

    if is_origin_model:
        log_file.write(
            "origin model PSNR_predicted=" + str(avg_psnr / iteration) + '\n')
        print("origin model PSNR_predicted=" + str(avg_psnr / iteration) + '\n')
    else:
        log_file.write(
            "new model PSNR_predicted=" + str(avg_psnr / iteration) + '\n')
        print("new model PSNR_predicted=" + str(avg_psnr / iteration) + '\n')
    return avg_psnr / iteration


opt = parser.parse_args()
cuda = opt.cuda

if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

image_list = glob.glob('../test/' + opt.dataset + "/*.*")
image_list = sorted(image_list)

avg_psnr_origin = 0.0
avg_psnr_new = 0.0


for img_id, image_name in enumerate(image_list, 1):
    if opt.trainset:
        check = os.path.join('../test/DIV2K_valid_LLR_small', image_name[-7:-4])
    else:
        check = os.path.join('../test/DIV2K_valid_LLR', image_name[-7:-4])
    if not os.path.exists(check):
        print("jump over")
        continue
    if opt.trainset:
        log_file = open('log_small.txt', "a")
    else:
        log_file = open('log.txt', "a")
    print("Processing ", image_name)
    # since image_name is '../test/LR/804.png'
    testloader = DataLoader(
        DataValSet_zssr('../test/', image_name[-7:-4]),
        batch_size=opt.batchSize, shuffle=False, pin_memory=False)

    origin_model = torch.load(opt.model)["model"]
    if cuda:
        model = origin_model.cuda()
    else:
        model = origin_model.cpu()

    # TODO: update model
    criterion = nn.MSELoss(size_average=True)
    if cuda:
        criterion = criterion.cuda()
        if opt.vgg_loss:
            netContent = netContent.cuda()

    print("===> Loading datasets")
    # filename = "../test/DIV2K_valid_LLR/DIV2K_LLR_" + str(img_id) + ".h5"
    # train_set = DatasetFromHdf5(filename)
    if opt.trainset:
        training_data_loader = DataLoader(DataValSet_zssr_train('../test/DIV2K_valid_LLR_small', image_name[-7:-4]),
                                          num_workers=opt.threads, \
                                          batch_size=opt.batchSize,
                                          shuffle=True)
    else:
        training_data_loader = DataLoader(DataValSet_zssr_train('../test/DIV2K_valid_LLR', image_name[-7:-4]),
                                          num_workers=opt.threads, \
                                          batch_size=opt.batchSize,
                                          shuffle=True)

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    if opt.trainset:
        result_dir = '../test/zssr_small/'
    else:
        result_dir = '../test/zssr/'
    print("===> Testing")
    log_file.write(image_name + '\n')
    avg_psnr_origin += test(testloader, origin_model, criterion, result_dir, log_file, True, image_name[-7:-4])

    print("===> Training")
    for epoch in range(1, opt.nEpochs + 1):
        train(training_data_loader, optimizer, model, criterion, epoch)

    print("===> Testing")
    log_file.write(image_name + '\n')
    avg_psnr_new += test(testloader, model, criterion, result_dir, log_file, False, image_name[-7:-4])
    log_file.close()

avg_psnr_origin = avg_psnr_origin / len(image_list)
avg_psnr_new = avg_psnr_new / len(image_list)

print("origin model PSNR_predicted=", avg_psnr_origin)
print("new model PSNR_predicted=", avg_psnr_new)
log_file = open('log.txt', "a")
log_file.write("origin model PSNR_predicted=" + str(avg_psnr_origin) + '\n')
log_file.write("new model PSNR_predicted=" + str(avg_psnr_new) + '\n')
log_file.close()
