import argparse, os
import torch
import torch.nn as nn
import numpy as np
from math import log10
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import DatasetFromFolder


parser = argparse.ArgumentParser(description="PyTorch SRResNet Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model/model_srresnet.pth", type=str, help="model path")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--batchSize", type=int, default=16, help="testing batch size")
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')


opt = parser.parse_args()
cuda = opt.cuda

if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

model = torch.load(opt.model)["model"]
criterion = nn.MSELoss()

image_dir = "../test/"
test_set = DatasetFromFolder(image_dir, opt.scale)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)

if cuda:
    model = model.cuda()
    criterion = criterion.cuda()

avg_psnr = 0.0
for batch in testing_data_loader:
    input, target = Variable(batch[0]), Variable(batch[1])
    if cuda:
        input = input.cuda()
        target = target.cuda()

    prediction = model(input)
    mse = criterion(prediction, target)
    psnr = 10 * log10(1 / mse.data[0])
    avg_psnr += psnr
print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))