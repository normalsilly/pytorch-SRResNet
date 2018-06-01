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

parser = argparse.ArgumentParser(description="PyTorch SRResNet Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model/model_srresnet.pth", type=str, help="model path") #TODO: change default model
parser.add_argument("--dataset", default="LR", type=str, help="dataset name, Default: LR")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--vgg_loss", action="store_true", help="Use content loss?")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")


def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)



opt = parser.parse_args()
cuda = opt.cuda
#eng = matlab.engine.start_matlab()

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
    im_gt_y = sio.loadmat(image_name)['label']
    im_l = sio.loadmat(image_name)['data']

    im_gt_y = im_gt_y.astype(float)
    im_l = im_l.astype(float)

    im_input = im_l.astype(np.float32).transpose(2,0,1)
    im_input = im_input.reshape(1,im_input.shape[0],im_input.shape[1],im_input.shape[2])
    im_input = Variable(torch.from_numpy(im_input/255.).float())

    if cuda:
        model = origin_model.cuda()
        im_input = im_input.cuda()
    else:
        model = origin_model.cpu()

    # TODO: update model
    criterion = nn.MSELoss(size_average=False)
    if cuda:
        criterion = criterion.cuda()
        if opt.vgg_loss:
            netContent = netContent.cuda()

    print("===> Loading datasets")
    filename = "../train/DIV2K_valid_LLR/DIV2K_LLR_" + img_id + ".h5"
    train_set = DatasetFromHdf5(filename)
    training_data_loader = DataLoader(dataset=train_set,
                                      num_workers=opt.threads, \
                                      batch_size=opt.batchSize,
                                      shuffle=True)

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    print("===> Training")
    train(training_data_loader, optimizer, model, criterion)

    HR_4x = origin_model(im_input)

    HR_4x = HR_4x.cpu()

    im_h = HR_4x.data[0].numpy().astype(np.float32)

    im_h = im_h*255.
    im_h = np.clip(im_h, 0., 255.)
    im_h = im_h.transpose(1,2,0).astype(np.float32)

    im_h_matlab = matlab.double((im_h / 255.).tolist())
    #im_h_ycbcr = eng.rgb2ycbcr(im_h_matlab)
    im_h_ycbcr = im_h_matlab
    im_h_ycbcr = np.array(im_h_ycbcr._data).reshape(im_h_ycbcr.size, order='F').astype(np.float32) * 255.
    im_h_y = im_h_ycbcr[:,:,0]

    psnr_predicted = PSNR(im_gt_y, im_h_y,shave_border=opt.scale)
    avg_psnr_predicted += psnr_predicted

    # New model
    HR_4x_new = model(im_input)

    HR_4x_new = HR_4x_new.cpu()

    im_h_new = HR_4x_new.data[0].numpy().astype(np.float32)

    im_h_new = im_h_new * 255.
    im_h_new = np.clip(im_h_new, 0., 255.)
    im_h_new = im_h_new.transpose(1, 2, 0).astype(np.float32)

    im_h_matlab_new = matlab.double((im_h_new / 255.).tolist())
    #im_h_ycbcr_new = eng.rgb2ycbcr(im_h_matlab_new)
    im_h_ycbcr_new = im_h_matlab_new
    im_h_ycbcr_new = np.array(im_h_ycbcr_new._data).reshape(im_h_ycbcr_new.size,
                                                    order='F').astype(
        np.float32) * 255.
    im_h_y_new = im_h_ycbcr_new[:, :, 0]

    psnr_predicted_new = PSNR(im_gt_y, im_h_y_new, shave_border=opt.scale)
    avg_psnr_predicted_new += psnr_predicted_new

    log_file.write(image_name + '\n')
    log_file.write("origin model PSNR_predicted=" + str(psnr_predicted) + '\n')
    log_file.write("new model PSNR_predicted=" + str(psnr_predicted_new) + '\n')

print("origin model PSNR_predicted=", avg_psnr_predicted/len(image_list))
print("new model PSNR_predicted=", avg_psnr_predicted_new/len(image_list))
# print("It takes average {}s for processing".format(avg_elapsed_time/len(image_list)))


# TODO: update lr
def train(training_data_loader, optimizer, model, criterion):

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

