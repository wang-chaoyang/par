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
from diffusers.optimization import get_scheduler

def build_env(args, logger):
    logging_dir = Path(args.env.o, args.env.log)
    accelerator_project_config = ProjectConfiguration(project_dir=args.env.o, logging_dir=logging_dir)
    deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=args.env.grad_accu)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.env.grad_accu,
        mixed_precision=args.env.mp,    
        log_with=args.env.report_to if not args.eval_only else None,
        project_config=accelerator_project_config,
        deepspeed_plugin=deepspeed_plugin if args.env.dpsd else None
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.env.seed is not None:
        set_seed(args.env.seed)

    if accelerator.is_main_process:
        if args.env.o is not None:
            folder = 'test' if args.eval_only else 'vis'
            os.makedirs(osp.join(args.env.o,folder), exist_ok=True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if args.env.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.env.scale_lr:
        args.optim.lr = (
            args.optim.lr * args.env.grad_accu * args.train.bs * accelerator.num_processes
        )
    

    return accelerator, weight_dtype



def build_optim(args, model, accelerator):
    assert args.optim.n=='adamw'
    optimizer_class = torch.optim.AdamW  
    optimizer = optimizer_class(
        filter(lambda p: p.requires_grad,model.parameters()),
        lr=args.optim.lr,
        betas=(args.optim.beta1, args.optim.beta2),
        weight_decay=args.optim.wd,
        eps=args.optim.eps,
    )

    assert args.env.its is not None
    lr_ratio = accelerator.num_processes    
    lr_scheduler = get_scheduler(
        args.lr_sch.n,
        optimizer=optimizer,
        num_warmup_steps=args.lr_sch.warm * lr_ratio,
        num_training_steps=args.env.its * lr_ratio,
    )

    return optimizer, lr_scheduler


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