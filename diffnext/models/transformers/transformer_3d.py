# ------------------------------------------------------------------------
# Copyright (c) 2024-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
# Modified in 2025 by Chaoyang Wang

from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from ..embeddings import PatchEmbed, PosEmbed, VideoPosEmbed, MaskEmbed, TextEmbed
from ..vision_transformer import VisionTransformer
from ..diffusion_mlp import DiffusionMLP
from einops import rearrange
import random

def pos_after_rot(x,rot,w):
    x_exp = x % w
    x_base = x - x_exp
    x_new = x_base + (x_exp+rot)%w
    return x_new


def mask_after_rot(x,rot,w):
    assert x.ndim==3
    x = rearrange(x,'b (h w) c -> b h w c',w=w)
    x = torch.concat([x[:,:,-rot:,:],x[:,:,:-rot,:]],dim=2)
    x = rearrange(x,'b h w c -> b (h w) c')
    return x

class Transformer3DModel(nn.Module):

    def __init__(
        self,
        video_encoder=None,
        image_encoder=None,
        image_decoder=None,
        mask_embed=None,
        text_embed=None,
        video_pos_embed=None,
        image_pos_embed=None,
        motion_embed=None,
        noise_scheduler=None,
        sample_schduler=None,
    ):
        super(Transformer3DModel, self).__init__()
        self.video_encoder:VisionTransformer = video_encoder
        self.image_encoder:VisionTransformer = image_encoder
        self.image_decoder:DiffusionMLP = image_decoder
        self.mask_embed:MaskEmbed = mask_embed
        self.text_embed:TextEmbed = text_embed
        self.video_pos_embed = video_pos_embed
        self.image_pos_embed = image_pos_embed
        self.motion_embed = motion_embed
        self.noise_scheduler = noise_scheduler
        self.sample_scheduler = sample_schduler
        self.pipeline_preprocess = lambda inputs: inputs
        self.loss_repeat = 4

    def progress_bar(self, iterable, enable=True):
        """Return a tqdm progress bar."""
        return tqdm(iterable) if enable else iterable

    def init_weights(self):
        """Initialze model weights."""
        [m.init_weights() if hasattr(m, "init_weights") else None for m in self.children()]

    def preprocess(self, inputs: Dict):
        """Preprocess model inputs."""
        dtype, device = self.dtype, self.device
        inputs["c"], add_guidance = inputs.get("c", []), inputs.get("guidance_scale", 1) != 1
        if inputs.get("x", None) is None:
            batch_size = inputs.get("batch_size", 1)
            image_size = (self.image_encoder.image_dim,) + self.image_encoder.image_size
            inputs["x"] = torch.empty(batch_size, *image_size, device=device, dtype=dtype)
        if inputs.get("prompt", None) is not None and self.text_embed:
            inputs["c"].append(self.text_embed(inputs.pop("prompt")))
        if inputs.get("motion_flow", None) is not None and self.motion_embed:
            flow, fps = inputs.pop("motion_flow", None), inputs.pop("fps", None)
            flow, fps = [v + v if (add_guidance and v) else v for v in (flow, fps)]
            inputs["c"].append(self.motion_embed(inputs["c"][-1], flow, fps))
        inputs["c"] = torch.cat(inputs["c"], dim=1) if len(inputs["c"]) > 1 else inputs["c"][0]

    @torch.no_grad()
    def postprocess(self, outputs: Dict, inputs: Dict):
        """Postprocess model outputs."""
        if inputs.get("output_type", "np") == "latent":
            return outputs
        x = inputs["vae"].unscale_(outputs.pop("x"))
        batch_size, vae_batch_size = x.size(0), inputs.get("vae_batch_size", 1)
        sizes, splits = [vae_batch_size] * (batch_size // vae_batch_size), []
        sizes += [batch_size - sum(sizes)] if sum(sizes) != batch_size else []
        for x_split in x.split(sizes) if len(sizes) > 1 else [x]:
            splits.append(inputs["vae"].decode(x_split).sample)
        x = torch.cat(splits) if len(splits) > 1 else splits[0]
        x = x.permute(0, 2, 3, 4, 1) if x.dim() == 5 else x.permute(0, 2, 3, 1)
        outputs["x"] = x.mul_(127.5).add_(127.5).clamp(0, 255).byte()

    def get_losses(self, z: torch.Tensor, x: torch.Tensor, video_shape=None) -> Dict:
        """Return the training losses."""
        z = z.repeat(self.loss_repeat, *((1,) * (z.dim() - 1)))
        x = x.repeat(self.loss_repeat, *((1,) * (x.dim() - 1)))
        x = self.image_encoder.patch_embed.patchify(x)
        noise = torch.randn(x.shape, dtype=x.dtype, device=x.device)
        timestep = self.noise_scheduler.sample_timesteps(z.shape[:2], device=z.device)
        x_t = self.noise_scheduler.add_noise(x, noise, timestep)
        x_t = self.image_encoder.patch_embed.unpatchify(x_t)
        timestep = getattr(self.noise_scheduler, "timestep", timestep)
        pred_type = getattr(self.noise_scheduler.config, "prediction_type", "flow")
        model_pred = self.image_decoder(x_t, timestep, z)
        model_target = noise.float() if pred_type == "epsilon" else noise.sub(x).float()
        loss = nn.functional.mse_loss(model_pred.float(), model_target, reduction="none")
        loss, weight = loss.mean(-1, True), self.mask_embed.mask.to(loss.dtype)
        weight = weight.repeat(self.loss_repeat, *((1,) * (z.dim() - 1)))
        loss = loss.mul_(weight).div_(weight.sum().add_(1e-5))
        if video_shape is not None:
            loss = loss.view((-1,) + video_shape).transpose(0, 1).sum((1, 2))
            i2i = loss[1:].sum().mul_(video_shape[0] / (video_shape[0] - 1))
            return {"loss_t2i": loss[0].mul(video_shape[0]), "loss_i2i": i2i}
        return {"loss": loss.sum()}

    @torch.no_grad()
    def denoise(self, z, x, guidance_scale=1, generator=None, pred_ids=None) -> torch.Tensor:
        """
        z: (b,k,d), output of AR
        x: (b,c,h,w / b,4,128,128), Gaussian noise 
        """
        """Run diffusion denoising process."""
        self.sample_scheduler._step_index = None  #
        for t in self.sample_scheduler.timesteps:
            x_pack = torch.cat([x] * 2) if guidance_scale > 1 else x
            timestep = torch.as_tensor(t, device=x.device).expand(z.shape[0])
            noise_pred = self.image_decoder(x_pack, timestep, z, pred_ids)  # 
            if guidance_scale > 1:
                cond, uncond = noise_pred.chunk(2)
                noise_pred = uncond.add_(cond.sub_(uncond).mul_(guidance_scale))
            noise_pred = self.image_encoder.patch_embed.unpatchify(noise_pred)  #
            x = self.sample_scheduler.step(noise_pred, t, x, generator=generator).prev_sample
        return self.image_encoder.patch_embed.patchify(x)

    @torch.inference_mode()
    def generate_frame(self, states: Dict, inputs: Dict):
        """Generate a batch of frames."""
        guidance_scale = inputs.get("guidance_scale", 1)
        min_guidance_scale = inputs.get("min_guidance_scale", guidance_scale)
        max_guidance_scale = inputs.get("max_guidance_scale", guidance_scale)
        generator = self.mask_embed.generator = inputs.get("generator", None)
        all_num_preds = [_ for _ in inputs["num_preds"] if _ > 0]
        guidance_end = max_guidance_scale if states["t"] else guidance_scale
        guidance_start = max_guidance_scale if states["t"] else min_guidance_scale
        c, x, self.mask_embed.mask = states["c"], states["x"].zero_(), None
        pos = self.image_pos_embed.get_pos(1, c.size(0)) if self.image_pos_embed else None
        for i, num_preds in enumerate(self.progress_bar(all_num_preds, inputs.get("tqdm2", False))):
            guidance_level = (i + 1) / len(all_num_preds)
            guidance_scale = (guidance_end - guidance_start) * guidance_level + guidance_start
            z = self.mask_embed(self.image_encoder.patch_embed(x)) 
            pred_mask, pred_ids = self.mask_embed.get_pred_mask(num_preds)  
            pred_ids = torch.cat([pred_ids] * 2) if guidance_scale > 1 else pred_ids
            prev_ids = prev_ids if i else pred_ids.new_empty((pred_ids.size(0), 0, 1))
            z = torch.cat([z] * 2) if guidance_scale > 1 else z
            z = self.image_encoder(z, c, prev_ids, pos=pos) # 
            prev_ids = torch.cat([prev_ids, pred_ids], dim=1)   #
            states["noise"].normal_(generator=generator)
            sample = self.denoise(z, states["noise"], guidance_scale, generator, pred_ids)
            x.add_(self.image_encoder.patch_embed.unpatchify(sample.mul_(pred_mask)))   #

    @torch.inference_mode()
    def generate_video(self, inputs: Dict):
        """Generate a batch of videos."""
        guidance_scale = inputs.get("guidance_scale", 1)
        max_latent_length = inputs.get("max_latent_length", 1)
        self.sample_scheduler.set_timesteps(inputs.get("num_diffusion_steps", 25))
        states = {"x": inputs["x"], "noise": inputs["x"].clone()}
        latents, self.mask_embed.pred_ids, time_pos = inputs.get("latents", []), None, []
        if self.image_pos_embed:
            time_pos = self.video_pos_embed.get_pos(max_latent_length).chunk(max_latent_length, 1)
        else:
            time_embed = self.video_pos_embed.get_time_embed(max_latent_length)
        [setattr(blk.attn, "cache_kv", True) for blk in self.video_encoder.blocks]
        for states["t"] in self.progress_bar(range(max_latent_length), inputs.get("tqdm1", True)):
            pos = time_pos[states["t"]] if time_pos else None
            c = self.video_encoder.patch_embed(states["x"])
            c.__setitem__(slice(None), self.mask_embed.bos_token) if states["t"] == 0 else c
            c = self.video_pos_embed(c.add_(time_embed[states["t"]])) if not time_pos else c
            c = torch.cat([c] * 2) if guidance_scale > 1 else c
            c = self.video_encoder(c, None if states["t"] else inputs["c"], pos=pos)
            states["c"] = self.video_encoder.mixer(states["*"], c) if states["t"] else c
            states["*"] = states["*"] if states["t"] else states["c"]
            if states["t"] == 0 and latents:
                states["x"].copy_(latents[-1])
            else:
                self.generate_frame(states, inputs)
                latents.append(states["x"].clone())
        [setattr(blk.attn, "cache_kv", False) for blk in self.video_encoder.blocks]

    def aux_loss(self, z: torch.Tensor, x: torch.Tensor, raw_noise=None, raw_timestep=None):
        """Return the training losses."""
        z = z.repeat(self.loss_repeat, *((1,) * (z.dim() - 1)))
        x = x.repeat(self.loss_repeat, *((1,) * (x.dim() - 1)))
        x = self.image_encoder.patch_embed.patchify(x)
        if raw_noise==None:
            assert raw_timestep==None
            noise = torch.randn(x.shape, dtype=x.dtype, device=x.device)
            timestep = self.noise_scheduler.sample_timesteps(z.shape[:2], device=z.device)
            raw_noise = noise.clone()
            raw_timestep = timestep.clone()
        else:
            noise = raw_noise
            timestep = raw_timestep
        x_t = self.noise_scheduler.add_noise(x, noise, timestep)
        x_t = self.image_encoder.patch_embed.unpatchify(x_t)
        timestep = getattr(self.noise_scheduler, "timestep", timestep)
        pred_type = getattr(self.noise_scheduler.config, "prediction_type", "flow")
        model_pred = self.image_decoder(x_t, timestep, z)
        model_target = noise.float() if pred_type == "epsilon" else noise.sub(x).float()
        loss = nn.functional.mse_loss(model_pred.float(), model_target, reduction="none")
        loss, weight = loss.mean(-1, True), self.mask_embed.mask.to(loss.dtype)
        weight = weight.repeat(self.loss_repeat, *((1,) * (z.dim() - 1)))
        loss = loss.mul_(weight).div_(weight.sum().add_(1e-5))
        return loss.sum(), raw_noise, raw_timestep, model_pred, model_target 


    def forward_train(self, inputs):

        inputs["x"].unsqueeze_(2) if inputs["x"].dim() == 4 else None
        bs, latent_length = inputs["x"].size(0), inputs["x"].size(2)
        c = self.video_encoder.patch_embed(inputs["x"][:, :, : latent_length - 1])
        bov = self.mask_embed.bos_token.expand(bs, 1, c.size(-2), -1)
        c = self.video_pos_embed(torch.cat([bov, c], dim=1))

        attn_mask, pos = self.mask_embed.get_attn_mask(c, inputs["c"]), None
        [setattr(blk.attn, "attn_mask", attn_mask) for blk in self.video_encoder.blocks]
        pos = self.video_pos_embed.get_pos(latent_length, bs) if self.image_pos_embed else pos
        c = self.video_encoder(c.flatten(1, 2), inputs["c"], pos=pos)
        if hasattr(self.video_encoder, "mixer") and latent_length > 1:
            c = c.view(bs, latent_length, -1, c.size(-1)).split([1, latent_length - 1], 1)
            c = torch.cat([c[0], self.video_encoder.mixer(*c)], 1)

        x = inputs["x"][:, :, :latent_length].transpose(1, 2).flatten(0, 1)

        raw_x = x.clone()  
        z = self.image_encoder.patch_embed(x)  
        pos = self.image_pos_embed.get_pos(1, bs) if self.image_pos_embed else pos
        _x,_prev_ids = self.mask_embed(z)
        z = self.image_encoder(_x, c.reshape(bs, -1, c.size(-1)), _prev_ids, pos=pos)

        loss1, raw_noise, raw_timestep, model_pred, model_target = self.aux_loss(z, x)
        h,w = self.config.image_base_size
        oh,ow = raw_x.shape[-2:]
        r = oh//h

        rot = random.randint(0,w-1)
        new_x = torch.concat([x[:,:,:,-rot*r:],x[:,:,:,:-rot*r]],dim=-1)

        self.mask_embed.mask = mask_after_rot(self.mask_embed.mask,rot,w)
        _prev_ids2 = pos_after_rot(_prev_ids,rot,w)
        new_noise = mask_after_rot(raw_noise,rot,w)
        new_timestep = mask_after_rot(raw_timestep[:,:,None],rot,w).squeeze(2)
    
        z2 = self.image_encoder.patch_embed(new_x)   
        pos = self.image_pos_embed.get_pos(1, bs) if self.image_pos_embed else pos
        _x2 = z2.mul(1 - self.mask_embed.mask).add_(self.mask_embed.mask_token * self.mask_embed.mask)
        z2 = self.image_encoder(_x2, c.reshape(bs, -1, c.size(-1)), _prev_ids2, pos=pos)
        loss2, _, _, model_pred2, model_target2 = self.aux_loss(z2, new_x, new_noise, new_timestep)

        model_pred1_rot = mask_after_rot(model_pred,rot,w)
        weight = self.mask_embed.mask.repeat(self.loss_repeat, *((1,) * (z.dim() - 1)))

        _loss2_aux = F.mse_loss(model_pred1_rot,model_pred2,reduction="none")
        loss2_aux = _loss2_aux.mul(weight).div(weight.sum().add(1e-5)).sum()
        loss = loss1 + 0.1*loss2_aux
        return {"loss": loss, "loss1":loss1, "loss2":loss2_aux}


    def forward(self, inputs):
        """Define the computation performed at every call."""
        self.pipeline_preprocess(inputs)
        self.preprocess(inputs)
        if self.training:
            return self.forward_train(inputs)
        inputs["latents"] = inputs.pop("latents", [])
        self.generate_video(inputs)
        return {"x": torch.stack(inputs["latents"], dim=2)}

