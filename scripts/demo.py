import os.path as osp
import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from diffnext.utils.builder import build_model
from diffnext.utils.val import DummyInferImage
import argparse


def main(args):

    accelerator = Accelerator()
    transformer, vae = build_model(args, accelerator)
    transformer.load_state_dict(torch.load(args.ckpt)['module'],strict=True)
    transformer.eval()
    pipe = DummyInferImage(transformer,vae,args)
    images = pipe(args.prompt, guidance_scale=args.cfg).images
    images[0].save('demo.jpg')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ptr',type=str,required=True)
    parser.add_argument('--ckpt',type=str,required=True)
    parser.add_argument('--prompt',type=str,required=True)
    parser.add_argument('--h',type=int,default=512)
    parser.add_argument('--w',type=int,default=1024)
    parser.add_argument('--zpad',type=int,default=8)
    parser.add_argument('--cfg',type=float,default=5)
    args = parser.parse_args()
    
    main(args)
