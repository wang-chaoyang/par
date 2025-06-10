import torch
from accelerate import Accelerator, DistributedType
import os
import os.path as osp
import warnings
import shutil
import torch.nn as nn

def save_normal(accelerator:Accelerator,args,logger,global_step,model):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
        save_path = os.path.join(args.env.o, f"checkpoint-{global_step}")
        accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.env.ckpt_num is not None:
            checkpoints = os.listdir(args.env.o)
            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

            if len(checkpoints) > args.env.ckpt_num:
                num_to_remove = len(checkpoints) - args.env.ckpt_num
                removing_checkpoints = checkpoints[0:num_to_remove]

                logger.info(
                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                )
                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                for removing_checkpoint in removing_checkpoints:
                    removing_checkpoint = os.path.join(args.env.o, removing_checkpoint)
                    shutil.rmtree(removing_checkpoint)
    
    return save_path



def resume_state(accelerator:Accelerator,args,upd_per_ep):

    first_epoch = 0
    resume_step = 0
    global_step = 0
    if args.resume:
        if args.resume != "latest":
            path = os.path.basename(args.resume)
        else:
            #
            dirs = os.listdir(args.env.o)
            dirsb = [d for d in dirs if d.startswith("best")]
            assert len(dirsb)<=1
            if len(dirsb)==0:
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            else:
                dirs = dirsb
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume}' does not exist. Starting a new training run."
            )
            args.resume = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            global_step = int(path.split("-")[1])
            resume_global_step = global_step * args.env.grad_accu
            first_epoch = global_step // upd_per_ep
            resume_step = resume_global_step % (upd_per_ep * args.env.grad_accu)
            accelerator.load_state(os.path.join(args.env.o, path))
    return first_epoch, resume_step, global_step


def loadweight(transformer:nn.Module, args):
    if isinstance(args.env.onlyw,str) and osp.isdir(args.env.onlyw):
        loaddir = args.env.onlyw
    else:
        loaddir = args.env.o
    dirs = os.listdir(loaddir)
    dirsb = [d for d in dirs if d.startswith("best")]
    assert len(dirsb)<=1
    if len(dirsb)==0:
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    else:
        dirs = dirsb
    path = dirs[-1] if len(dirs) > 0 else None

    if path is not None:
        loadpath = osp.join(loaddir,path)
        print(f'load ckpt from {loadpath}')
        state_dict = torch.load(osp.join(loaddir,path,'pytorch_model/mp_rank_00_model_states.pt'))['module']
        if hasattr(transformer,'module'):
            transformer.module.load_state_dict(state_dict,strict=True)
        else:
            transformer.load_state_dict(state_dict['module'],strict=True)