import torch.utils.data as data
import torch
import h5py

from os import listdir
from os.path import join
from skimage.io import imread
# from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


# def input_transform(crop_size, upscale_factor):
#     return Compose([
#         CenterCrop(crop_size),
#         Resize(crop_size // upscale_factor),
#         ToTensor(),
#     ])


def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


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


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, upscale_factor=4):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]

        # self.input_transform = input_transform(crop_size, upscale_factor)
        self.target_transform = target_transform(crop_size)

    def __getitem__(self, index):
        input = imread(self.image_filenames[index])
        target = input.copy()
        # if self.input_transform:
        #     input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)