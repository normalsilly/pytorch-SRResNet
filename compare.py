import argparse, os
import torch
import torch.nn as nn
import numpy as np
import time
import math
import pandas as pd
import scipy.misc
from math import log10
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import DataValSet_test
from torchvision import transforms
from os.path import join
from srresnet import _NetC
from pdb import set_trace

DATA_LIST_PATH = '../test/val.txt'

parser = argparse.ArgumentParser(description="PyTorch SRResNet Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model/model_srresnet.pth", type=str, help="model path")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--batchSize", type=int, default=1, help="testing batch size")
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")


def eval(test_gen, model, criterion, SR_dir):
    avg_time = 0
    # set_trace()
    for iteration, batch in enumerate(test_gen, 1):
        input_low, input, target = Variable(batch[0], volatile=True), Variable(batch[1], volatile=True), Variable(batch[2], volatile=True)
        input_low = input_low.cuda()
        input = input.cuda()
        target = target.cuda()

        start = time.clock()
        Blur_SR_low = model(input_low)
        Blur_SR = model(input)

        im_h = Blur_SR.cpu().data[0].numpy().astype(np.float32)

        im_h_low = Blur_SR_low.cpu().data[0].numpy().astype(np.float32)
        avg_time += (time.clock() - start)

        SR = Variable((torch.from_numpy(im_h)).unsqueeze(0)).cuda()
        SR_low = Variable((torch.from_numpy(im_h_low)).unsqueeze(0)).cuda()

        result = transforms.ToPILImage()(SR.cpu().data[0])
        result_low = transforms.ToPILImage()(SR_low.cpu().data[0])
        path = join(SR_dir, '{0:04d}_RE.jpg'.format(iteration))
        path_low = join(SR_dir, 'low_{0:04d}_RE.jpg'.format(iteration))
        result.save(path)
        result_low.save(path_low)
        if iteration < 5:
            difference_LR = input - SR_low
            difference_LR = difference_LR.cpu().data[0].numpy().astype(np.float32)
            difference_SR = target - SR
            difference_SR = difference_SR.cpu().data[0].numpy().astype(np.float32)
            print(difference_LR)
            print(difference_LR.shape)
            save_to_csv('difference_LR_' + str(iteration) + '_0.csv', difference_LR[0])
            save_to_csv('difference_LR_' + str(iteration) + '_1.csv', difference_LR[1])
            save_to_csv('difference_LR_' + str(iteration) + '_2.csv', difference_LR[2])
            LR_size = difference_LR.shape
            difference_SR = scipy.misc.imresize(difference_SR, LR_size, interp='bicubic', mode=None)
            print('difference_SR')
            print(difference_SR.shape)
            print(difference_SR)
            save_to_csv('difference_SR_' + str(iteration) + '_0.csv', difference_SR[0])
            save_to_csv('difference_SR_' + str(iteration) + '_1.csv', difference_SR[1])
            save_to_csv('difference_SR_' + str(iteration) + '_2.csv', difference_SR[2])


    print("===> Avg. Time: {:.4f} s".format(avg_time / iteration))


def save_to_csv(filename, data):
    df = pd.DataFrame(data)
    df.to_csv(filename, header=False, index=False)

opt = parser.parse_args()
cuda = opt.cuda

if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

print("===> Loading datasets")
root_val_dir = '../test/'
SR_dir = join(root_val_dir, 'Compare_Results')
if os.path.isdir(SR_dir):
    pass
else:
    os.mkdir(SR_dir)

testloader = DataLoader(DataValSet_test(root_val_dir, DATA_LIST_PATH), batch_size=opt.batchSize, shuffle=False, pin_memory=False)
model = _NetC()
criterion = torch.nn.MSELoss(size_average=True)

if cuda:
    model = model.cuda()
    criterion = criterion.cuda()

if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        opt.start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model"].state_dict())
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))

# test(testloader, model, criterion, SR_dir)

eval(testloader, model, criterion, SR_dir)


