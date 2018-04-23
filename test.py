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


DATA_LIST_PATH = '../test/val.txt'

parser = argparse.ArgumentParser(description="PyTorch SRResNet Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model/model_srresnet.pth", type=str, help="model path")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--batchSize", type=int, default=1, help="testing batch size")
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')


def eval(test_gen, model, criterion, SR_dir):
    avg_psnr = 0
    avg_psnr_low = 0
    avg_time = 0
    for iteration, batch in enumerate(test_gen, 1):
        input_low, input, target = Variable(batch[0], volatile=True), Variable(batch[1], volatile=True), Variable(batch[2], volatile=True)
        input_low = input_low.cuda()
        input = input.cuda()
        target = target.cuda()

        start = time.clock()
        Blur_SR_low = model(input_low)
        Blur_SR = model(input)

        im_h = Blur_SR.cpu().data[0].numpy().astype(np.float32)
        im_h[im_h < 0] = 0
        im_h[im_h > 1.] = 1.

        im_h_low = Blur_SR_low.cpu().data[0].numpy().astype(np.float32)
        im_h_low[im_h_low < 0] = 0
        im_h_low[im_h_low > 1.] = 1.
        avg_time += (time.clock() - start)

        SR = Variable((torch.from_numpy(im_h)).unsqueeze(0)).cuda()
        SR_low = Variable((torch.from_numpy(im_h_low)).unsqueeze(0)).cuda()

        result = transforms.ToPILImage()(SR.cpu().data[0])
        result_low = transforms.ToPILImage()(SR_low.cpu().data[0])
        path = join(SR_dir, '{0:04d}_RE.jpg'.format(iteration))
        path_low = join(SR_dir, '{0:04d}_RE_low.jpg'.format(iteration))

        p = join('../test/', 'Origin')
        p1 = join(p, 'LR/{0:04d}.jpg'.format(iteration))
        p2 = join(p, 'HR/{0:04d}.jpg'.format(iteration))
        originLR = transforms.ToPILImage()(input.cpu().data[0])
        originHR = transforms.ToPILImage()(target.cpu().data[0])
        originLR.save(p1)
        originHR.save(p2)
        result.save(path)
        result_low.save(path_low)

        mse = criterion(SR, target)
        mse_low = criterion(SR_low, input)
        psnr = 10 * log10(1 / mse.data[0])
        psnr_low = 10 * log10(1 / mse_low.data[0])
        avg_psnr += psnr
        avg_psnr_low += psnr_low
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
        # print(iteration)
        # print(psnr)
        # print(psnr_low)

    print("===> Avg. SR PSNR: {:.4f} dB".format(avg_psnr / iteration))
    print("===> Avg. LR PSNR: {:.4f} dB".format(avg_psnr_low / iteration))
    print("===> Avg. Time: {:.4f} s".format(avg_time / iteration))


def save_to_csv(filename, data):
    df = pd.DataFrame(data)
    df.to_csv(filename, header=False, index=False)


def test(test_gen, model, criterion, SR_dir):
    avg_psnr = 0
    avg_psnr2 = 0
    avg_time = 0

    for iteration, batch in enumerate(test_gen, 1):
        input,  target = Variable(batch[0], volatile=True), Variable(batch[1], volatile=True)
        input = input.cuda()
        target = target.cuda()

        start = time.clock()
        Blur_SR = model(input)

        im_h = Blur_SR.cpu().data[0].numpy().astype(np.float32)
        im_h[im_h < 0] = 0
        im_h[im_h > 1.] = 1.
        avg_time += (time.clock() - start)

        SR = Variable((torch.from_numpy(im_h)).unsqueeze(0)).cuda()

        result = transforms.ToPILImage()(SR.cpu().data[0])
        path = join(SR_dir, '{0:04d}_RE.jpg'.format(iteration))
        result.save(path)
        mse = criterion(SR, target)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
        mse2 = criterion(Blur_SR, target)
        psnr2 = 10 * log10(1 / mse2.data[0])
        avg_psnr2 += psnr2
        print(iteration)
        print(psnr)
        print(psnr2)

    print("===> Avg. SR PSNR: {:.4f} dB".format(avg_psnr / iteration))
    print("===> Avg. SR PSNR2: {:.4f} dB".format(avg_psnr2 / iteration))
    print("===> Avg. Time: {:.4f} s".format(avg_time / iteration))

opt = parser.parse_args()
cuda = opt.cuda

if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

print("===> Loading datasets")
root_val_dir = '../test/'
SR_dir = join(root_val_dir, 'Results')
if os.path.isdir(SR_dir):
    pass
else:
    os.mkdir(SR_dir)

testloader = DataLoader(DataValSet_test(root_val_dir, DATA_LIST_PATH), batch_size=opt.batchSize, shuffle=False, pin_memory=False)
model = torch.load(opt.model)["model"]
criterion = torch.nn.MSELoss(size_average=True)

if cuda:
    model = model.cuda()
    criterion = criterion.cuda()

# test(testloader, model, criterion, SR_dir)

eval(testloader, model, criterion, SR_dir)


