import sys
sys.path.append('.')

import math
import os
import numpy as np
import torch
import torch.utils.checkpoint
from accelerate.logging import get_logger

from tqdm import tqdm
from omegaconf import OmegaConf
import diffusers
from diffnext.data.pre_dl import pr_train_dl, pr_val_dl
from diffnext.utils.builder import build_env, build_optim, build_model
from diffnext.utils.hook import save_normal, resume_state, loadweight
from diffnext.utils.val import val_fn
from diffnext.utils.utils import encode_image


logger = get_logger(__name__)

def main(args):

    train_dataloader = pr_train_dl(args)
    val_dataloader = pr_val_dl(args)
    accelerator, weight_dtype = build_env(args, logger)
    transformer, vae = build_model(args, accelerator)
    optimizer, lr_scheduler = build_optim(args,transformer,accelerator)
    transformer, optimizer, train_dataloader, lr_scheduler, val_dataloader = accelerator.prepare(
            transformer, optimizer, train_dataloader, lr_scheduler, val_dataloader)

    overrode_max_train_steps = False
    upd_per_ep = math.ceil(len(train_dataloader) / args.env.grad_accu)
    if overrode_max_train_steps:
        args.env.its = args.eps * upd_per_ep
    args.eps = math.ceil(args.env.its / upd_per_ep)

    if accelerator.is_main_process:
        accelerator.init_trackers("model")

    gstep = 0
    f_ep = 0
    resume_step = 0

    if args.env.onlyw:
        loadweight(transformer, args)
    else:
        f_ep, resume_step, gstep = resume_state(accelerator,args,upd_per_ep)
    torch.cuda.empty_cache()
    progress_bar = tqdm(range(gstep, args.env.its), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    if args.eval_only:
        val_fn(
            accelerator,
            args,
            vae,
            transformer,
            val_dataloader,
            weight_dtype,
            gstep=gstep,
            )
    else:
        for ep in range(f_ep, args.eps):
            train_loss = 0.0
            for step, batch in enumerate(train_dataloader):
                transformer.train()
                if args.resume and ep == f_ep and step < resume_step:
                    if step % args.env.grad_accu == 0:
                        progress_bar.update(1)
                    continue

                with accelerator.accumulate(transformer):
                    
                    img = batch['img']
                    inputs = {}
                    inputs["raw_x"] = img  
                    h,w = transformer.module.config.image_base_size if hasattr(transformer,'module') else transformer.config.image_base_size
                    oh, ow = img.shape[-2:]
                    dr = oh//h  
                    vae_dr = dr // 2
                    
                    x = encode_image(img,vae,args,vae_dr)
                    inputs["x"] = x.to(weight_dtype)

                    bsz = x.shape[0]
                    with torch.no_grad():
                        c = transformer.text_embed(batch['text'])
                        inputs["c"] = [c]

                    all_loss = transformer.forward(inputs)
                    loss, loss1, loss2 = all_loss['loss'], all_loss['loss1'], all_loss['loss2']
                    
                    avg_loss = accelerator.gather(loss.repeat(args.train.bs)).mean()
                    train_loss += avg_loss.item() / args.env.grad_accu
                    train_loss1 = accelerator.gather(loss1.repeat(args.train.bs)).mean().item() / args.env.grad_accu
                    train_loss2 = accelerator.gather(loss2.repeat(args.train.bs)).mean().item() / args.env.grad_accu

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(transformer.parameters(), args.env.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    gstep += 1
                    accelerator.log({"all loss": train_loss, 
                                     "diffusion loss":train_loss1, 
                                     "aux loss":train_loss2, 
                                     }, step=gstep)
                    train_loss = 0.0

                    if gstep % args.env.sv_its == 0:
                        save_normal(accelerator,args,logger,gstep,transformer)

                logs = {"step_loss": loss.detach().item()}
                progress_bar.set_postfix(**logs)

                if gstep >= args.env.its:
                    accelerator.wait_for_everyone()
                    break

        accelerator.end_training()


if __name__ == "__main__":
    cfg_path = sys.argv[1]
    assert os.path.isfile(cfg_path)
    args = OmegaConf.load(cfg_path)
    cli_config = OmegaConf.from_cli(sys.argv[2:])
    args = OmegaConf.merge(args,cli_config)
    main(args)
