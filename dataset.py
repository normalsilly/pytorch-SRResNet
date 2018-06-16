import torch.utils.data as data
import torch
import h5py
import numpy as np
import random
import os

from os import listdir
from os.path import join
from skimage.io import imread
# from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path)
        self.data = hf.get("data")
        self.target = hf.get("label")

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index,:,:,:]).float(), torch.from_numpy(self.target[index,:,:,:]).float()

    # def __getitem__(self, index):
    #     # randomly flip
    #     # print(index)
    #     # data shppe: C*H*W
    #     LR_patch = self.data[index, :, :, :]
    #     HR_patch = self.target[index, :, :, :]
    #     LR_patch = np.clip(LR_patch, 0,
    #                        1)  # we might get out of bounds due to noise
    #     HR_patch = np.clip(HR_patch, 0,
    #                        1)  # we might get out of bounds due to noise
    #     LR_patch = np.asarray(LR_patch, np.float32)
    #     HR_patch = np.asarray(HR_patch, np.float32)
    #
    #     flip_channel = random.randint(0, 1)
    #     if flip_channel != 0:
    #         LR_patch = np.flip(LR_patch, 2)
    #         HR_patch = np.flip(HR_patch, 2)
    #     # randomly rotation
    #     rotation_degree = random.randint(0, 3)
    #     LR_patch = np.rot90(LR_patch, rotation_degree, (1, 2))
    #     HR_patch = np.rot90(HR_patch, rotation_degree, (1, 2))
    #     return LR_patch.copy(), \
    #            HR_patch.copy()

    def __len__(self):
        return self.data.shape[0]


class DataValSet_test(data.Dataset):
    def __init__(self, root, list_path, mean=(128, 128, 128)):
        self.root = root
        self.list_path = list_path
        self.mean = mean
        self.img_ids = [i_id.strip() for i_id in open(list_path)]

        self.files = []
        for name in self.img_ids:
            LLR_file = join(self.root, "LLR/%s.png" % name)
            LR_file = join(self.root, "LR/%s.png" % name)
            HR_file = join(self.root, "HR/%s.png" % name)
            self.files.append({
                "llr": LLR_file,
                "lr": LR_file,
                "hr": HR_file
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        LLR_image = imread(datafiles["llr"])
        LLR_image = LLR_image.transpose((2, 0, 1))
        LLR_image = np.asarray(LLR_image, np.float32)
        LLR_image /= 255

        LR_image = imread(datafiles["lr"])
        LR_image = LR_image.transpose((2, 0, 1))
        LR_image = np.asarray(LR_image, np.float32)
        LR_image /= 255

        HR_image = imread(datafiles["hr"])
        HR_image = HR_image.transpose((2, 0, 1))
        HR_image = np.asarray(HR_image, np.float32)
        HR_image /= 255
        return LLR_image.copy(), LR_image.copy(), HR_image.copy()

# TODO: check image_name
class DataValSet_zssr(data.Dataset):
    def __init__(self, root, image_name, mean=(128, 128, 128)):
        self.root = root
        self.mean = mean
        self.files = []
        LR_file = join(self.root, "LR/%s.png" % image_name)
        HR_file = join(self.root, "HR/%s.png" % image_name)
        self.files.append({
            "lr": LR_file,
            "hr": HR_file
        })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        LR_image = imread(datafiles["lr"])
        LR_image = LR_image.transpose((2, 0, 1))
        LR_image = np.asarray(LR_image, np.float32)
        LR_image /= 255

        HR_image = imread(datafiles["hr"])
        HR_image = HR_image.transpose((2, 0, 1))
        HR_image = np.asarray(HR_image, np.float32)
        HR_image /= 255
        return LR_image.copy(), HR_image.copy()

class DataValSet_zssr_train(data.Dataset):
    def __init__(self, root, image_name):
        super(DataValSet_zssr_train, self).__init__()
        self.root = root
        self.input_dir = os.path.join(self.root, image_name, 'input_4x')
        self.target_dir = os.path.join(self.root, image_name, 'target')

        # for split in ["train", "trainval", "val"]:
        self.input_ids = [x for x in sorted(os.listdir(self.input_dir)) if is_image_file(x)]

    def __getitem__(self, index):
        # randomly flip
        #print(index)
        #data shppe: C*H*W
        name = self.input_ids[index]
        input_image = imread(os.path.join(self.input_dir, "%s" % name))
        input_image = input_image.transpose((2, 0, 1))
        input_image = np.asarray(input_image, np.float32)
        input_image /= 255

        target_image = imread(os.path.join(self.target_dir, "%s" % name))
        target_image = target_image.transpose((2, 0, 1))
        target_image = np.asarray(target_image, np.float32)
        target_image /= 255

        flip_channel = random.randint(0, 1)
        if flip_channel != 0:
            input_image = np.flip(input_image, 2)
            target_image = np.flip(target_image, 2)
        # randomly rotation
        rotation_degree = random.randint(0, 3)
        input_image = np.rot90(input_image, rotation_degree)
        target_image = np.rot90(target_image, rotation_degree)
        return input_image.copy(), \
               target_image.copy()

    def __len__(self):
        return len(self.input_ids)
