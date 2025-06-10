import torch
import torchvision.transforms as T
import numpy as np

def normalize_01_into_pm1(x):
    return x.add(x).add_(-1)

def shift_panorama(x:torch.Tensor):
    assert x.ndim==3
    r = np.random.randint(0,x.shape[-1])
    x_ = torch.concat([x[:,:,r:],x[:,:,:r]],dim=-1)
    return x_


def get_train_transform(args):
    transform = [
            T.ToTensor(), 
            normalize_01_into_pm1
        ]
    if args.aug.shift==True:
        transform.append(shift_panorama)
    return T.Compose(transform)

def get_val_transform():
    transform = [
            T.ToTensor(), 
            normalize_01_into_pm1
        ]
    return T.Compose(transform)