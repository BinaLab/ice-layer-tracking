from base import BaseDataSet, BaseDataLoader
from utils import palette
from glob import glob
import numpy as np
import os
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class RDSDataset(BaseDataSet):
    def __init__(self, **kwargs):
        # self.num_classes = 3
        # self.num_classes = 28 # @dv: of snow radar
        self.num_classes = 252 # @dv: for rds
        # self.num_classes = 30
        # self.palette = palette.CReSIS_palette
        # self.palette = palette.CReSIS_palette_new
        # self.palette = palette.CityScpates_palette
        # self.palette = palette.Viridis_palette
        self.palette = palette.CReSIS_palette_final
        super(RDSDataset, self).__init__(**kwargs)

    def _set_files(self):
        if self.split in ['train', 'val', 'train_28', 'val_28']:
            file_list = os.path.join(self.root, self.split + '_pair_semantic.txt')
            self.files = [name.rstrip() for name in tuple(open(file_list, "r"))]
        else: raise ValueError(f"Invalid split name {self.split} choose one of [train, val]")

    def _load_data(self, index):
        image_id = self.files[index] # assuming image_id is a string 
        image_path, label_path = image_id.split()
        image_path = os.path.join(self.root,image_path)
        label_path = os.path.join(self.root,label_path)
        # image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32) # @dv: had been using this till now
        # label = np.asarray(Image.open(label_path), dtype=np.int32) # @dv: had been using this till now
        image = cv2.imread(image_path, -1) # @dv: copied from dr. yari's code on 06/25/20
        if image is None:
            print(image_path)
        ### @dv: exp1-10/4/2020 ###
        image[image<0] = 0
        image = np.float32(np.uint8(255*image))
        ### @dv: exp1-10/4/2020 ###
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB) # @dv
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE) # @dv: copied from dr. yari's code on 06/25/20
        return image, label, image_id



class RDS(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, val=False,
                    shuffle=False, flip=False, rotate=False, blur= False, augment=True, val_split= None, return_id=False,
                    drop_last=False): # drop_last added here by @dv # augment changed from False to True by @dv

        # self.MEAN = [0.28689529, 0.32513294, 0.28389176] # @dv: author's default?
        # self.MEAN = [144.5356690370986] # @dv: for cresis
        self.MEAN = [0.485, 0.456, 0.406] # @dv: Imagenet's mean
        # self.MEAN = [0.7709920515555939, 0.7709920515555939, 0.7709920515555939] # @dv: RDS' globally min-max normalized mean
        
        # self.STD = [0.17613647, 0.18099176, 0.17772235] # @dv: author's default?
        # self.STD = [32.806646309835145] # @dv: for cresis
        self.STD = [0.229, 0.224, 0.225] # @dv: Imagenet's std
        # self.STD = [0.065698269476182, 0.065698269476182, 0.065698269476182] # @dv: RDS' globally min-max normalized std

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val,
            'drop_last': drop_last # by @dv
        }

        self.dataset = RDSDataset(**kwargs)
        super().__init__(self.dataset, batch_size, shuffle, num_workers, drop_last, val_split) # @dv: including drop_last
        # super().__init__(self.dataset, batch_size, shuffle, num_workers, val_split) # @dv: error fix
        # super(CReSIS, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split, drop_last) # original @dv


