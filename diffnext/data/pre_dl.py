import torch
from torch.utils.data import DataLoader
from .transform import get_train_transform, get_val_transform
from .matter_recap import MatterDatasetRecap


def collate_fn(batch):
    imgs = []
    texts = []
    metas = []
    for sample in batch:
        imgs.append(sample['img'])
        texts.append(sample['text'])
        metas.append(sample['meta'])
    return dict(img=torch.stack(imgs,0), text=texts, meta=metas)


def pr_train_dl(p):
    
    train_tf = get_train_transform(p)
    train_dataset = MatterDatasetRecap(
        path = p.path,
        text_path=p.textpath,
        split = 'train',
        size = (p.h,p.w),
        transform = train_tf
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=p.train.bs,
        num_workers=p.train.workers,
        shuffle=True,  #
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    return train_dataloader

def pr_val_dl(p):

    val_tf = get_val_transform()
    val_dataset = MatterDatasetRecap(
        path = p.path,
        text_path=p.textpath,
        split = 'test',
        size =(p.h,p.w),
        transform=val_tf
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=p.eval.bs,
        num_workers=p.eval.workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    return val_dataloader