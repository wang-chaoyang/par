import os
import os.path as osp
import torch
import torch.nn as nn
from pathlib import Path
from accelerate.utils import ProjectConfiguration, set_seed, DeepSpeedPlugin
from accelerate import Accelerator
import logging
import transformers
import diffusers
from omegaconf import OmegaConf
from diffusers.optimization import get_scheduler
import datetime

def build_model(args, accelerator):

    from diffnext.models.transformers.transformer_nova import NOVATransformer3DModel
    from diffnext.models.text_encoders.phi import PhiEncoderModel
    from diffnext.models.autoencoders.autoencoder_kl import AutoencoderKL
    from diffnext.models.autoencoders.autoencoder_kl_opensora import AutoencoderKLOpenSora
    from diffnext.schedulers.scheduling_flow import FlowMatchEulerDiscreteScheduler
    from diffnext.schedulers.scheduling_ddpm import DDPMScheduler
    from transformers import CodeGenTokenizerFast
    from diffusers.loaders import PeftAdapterMixin

    model_id = args.ptr

    vae_class = AutoencoderKL
    scheduler_class = FlowMatchEulerDiscreteScheduler
    
    class NOVATransformer_peft(NOVATransformer3DModel, PeftAdapterMixin): pass

    device = accelerator.device
    
    # if args.get('msz',None) is None:
    #     args.msz = args.sz

    tf_cfg = dict(video_base_size=[1,args.h//32, args.w//32],
                  image_base_size=[args.h//16, args.w//16],
                  image_size=(args.h, args.w))
    transformer = NOVATransformer_peft.from_pretrained(f"{model_id}/transformer",**tf_cfg).to(device)
    vae:nn.Module = vae_class.from_pretrained(f"{model_id}/vae").to(device)
    text_encoder = PhiEncoderModel.from_pretrained(f"{model_id}/text_encoder").to(device)
    tokenizer = CodeGenTokenizerFast.from_pretrained(f"{model_id}/tokenizer")
    noise_scheduler = scheduler_class.from_pretrained(f"{model_id}/scheduler")
    sample_scheduler = scheduler_class.from_pretrained(f"{model_id}/scheduler")
    
    text_encoder.requires_grad_(False)
    transformer.text_embed.encoders = [tokenizer, text_encoder]
    transformer.noise_scheduler = noise_scheduler
    transformer.sample_scheduler = sample_scheduler

    vae.requires_grad_(False)
    

    return transformer, vae