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
from dataset import DataValSet_zssr
from torchvision import transforms
from os.path import join
from math import log10


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

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr


# TODO: update lr
def train(training_data_loader, optimizer, model, criterion):
    lr = adjust_learning_rate(optimizer, 1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = 1e-4

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


def test(test_gen, model, criterion, SR_dir, log_file, is_origin_model):
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
            path = join(SR_dir, '{0:04d}_origin.jpg'.format(iteration))
        else:
            path = join(SR_dir, '{0:04d}_new.jpg'.format(iteration))
        result.save(path)
        mse = criterion(SR, target)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
        # print(iteration)
        # print(psnr)

    if is_origin_model:
        log_file.write(
            "origin model PSNR_predicted=" + str(avg_psnr / iteration) + '\n')
        print("origin model PSNR_predicted=" + str(avg_psnr / iteration) + '\n')
    else:
        log_file.write(
            "new model PSNR_predicted=" + str(avg_psnr / iteration) + '\n')
        print("new model PSNR_predicted=" + str(avg_psnr / iteration) + '\n')


opt = parser.parse_args()
cuda = opt.cuda

with open('n.txt', 'r') as f:
    tmp_n = f.readlines()
n = []
for i in tmp_n:
    n.append(int(i[:-1]))

log_file = open('log.txt', "w")

if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

origin_model = torch.load(opt.model)["model"]

image_list = glob.glob('../test/' + opt.dataset + "/*.*")
image_list = sorted(image_list)

avg_psnr_predicted = 0.0
avg_psnr_predicted_new = 0.0
avg_elapsed_time = 0.0


for img_id, image_name in enumerate(image_list, 1):
    print("Processing ", image_name)
    if img_id in n:
        print('jump over')
        continue
    testloader = DataLoader(
        DataValSet_zssr('../test/' + opt.dataset + '/', image_name),
        batch_size=opt.batchSize, shuffle=False, pin_memory=False)

    if cuda:
        model = origin_model.cuda()
    else:
        model = origin_model.cpu()

    # TODO: update model
    criterion = nn.MSELoss(size_average=False)
    if cuda:
        criterion = criterion.cuda()
        if opt.vgg_loss:
            netContent = netContent.cuda()

    print("===> Loading datasets")
    filename = "../test/DIV2K_valid_LLR/DIV2K_LLR_" + str(img_id) + ".h5"
    train_set = DatasetFromHdf5(filename)
    training_data_loader = DataLoader(dataset=train_set,
                                      num_workers=opt.threads, \
                                      batch_size=opt.batchSize,
                                      shuffle=True)

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    print("===> Training")
    train(training_data_loader, optimizer, model, criterion)

    print("===> Testing")
    log_file.write(image_name + '\n')
    test(testloader, model, criterion, '../test/zssr/', log_file, True)
    test(testloader, origin_model, criterion, '../test/zssr/', log_file, False)

print("origin model PSNR_predicted=", avg_psnr_predicted/len(image_list))
print("new model PSNR_predicted=", avg_psnr_predicted_new/len(image_list))
