import os
import numpy as np
from PIL import Image
import torch
from torch.utils import data
from .loader_utils import read_image_list

class thermalLoader(data.Dataset):
    
    mean_rgb = [0.0, 0.0, 0.0]

    def __init__(
        self,
        image_path,
        label_path,
        img_list='',
        img_size=(512, 640),
        n_channels = 3,
        img_normalize=True,
    ):
        """
        Parameters: 
            -   image_path: the path to images
            -   label_path: the path to labels
            -   image_list: A path to the file that contains the  list of images for the loader
            -   img_size: image dimensions
            -   n_channels: number of channels (RGB etc.)
            -   img_normalize: a boolean value indicating if the images should be normalized
        """

        self.image_path = image_path
        self.label_path = label_path
        self.img_list = img_list
        self.img_normalize = img_normalize


        self.n_channels = n_channels
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.file_prefixes = []
        self.split_prefixes = []
        self.split_prefixes, self.file_prefixes, self.file_extensions = read_image_list(self.img_list)
        


    def __len__(self):
        return len(self.file_prefixes)


    def __getitem__(self, index):
        prefix = self.file_prefixes[index]
        split_dir = self.split_prefixes[index]
        extension = self.file_extensions[index]
        img_path = os.path.join(self.image_path, prefix + '.' + extension)
        img = Image.open(img_path)
        img = np.array(img, dtype=np.uint8)
        label_path = os.path.join(self.label_path, prefix + '.' + extension)
        label = Image.open(label_path)
        label = np.array(label, dtype=np.uint8)

        # change label pixels with value 255 to 13
        mask = (label == 255)
        label[mask] = 13


        img = img.astype(np.float) / 255.0
        if img.ndim > 2:
            img = img.transpose(2, 0, 1)            # HWC -> CHW
        else:
            img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img).float()
        label = torch.from_numpy(label).long()


        if self.img_normalize is True:
            for i in range (0,self.n_channels):
                img[i,:,:] -= (self.mean_rgb[i])

        return img, label, self.file_prefixes[index]


    def get_img_prefix(self, index):
        return self.file_prefixes[index]


