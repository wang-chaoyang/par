import os
import os.path as osp
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms as T


class MatterDatasetRecap(Dataset):
    def __init__(self, path, text_path, split, size, transform=None):
        super().__init__()
        self.path = path
        self.text_path = text_path
        self.size = size
        self.data = []
        filenames = np.load(osp.join(path,f'{split}.npy'))
        for _d in filenames:
            name = _d[0].split('_skybox0_sami')[0].split('/')
            self.data.append((name[0],name[2]))
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        id1,id2 = self.data[idx]
        img = Image.open(osp.join(self.path,id1,'matterport_stitched_images',f'{id2}.png')).convert('RGB').resize(size=self.size[::-1],resample=Image.Resampling.BILINEAR)
        with open(osp.join(self.text_path,f'{id1}_{id2}.txt'),'r',encoding='utf-8') as f: text = f.readline() 
        return dict(img=self.transform(img),text=text,meta=dict(id1=id1,id2=id2))



if __name__ == '__main__':
    pass


