import torch.utils.data as data
import torch
import h5py
import numpy as np

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
