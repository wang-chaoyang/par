import torch
from diffnext.pipelines import NOVAPipeline
from diffnext.pipelines.nova.pipeline_utils import NOVAPipelineOutput
import copy
import os
import os.path as osp
import numpy as np
from typing import List
from PIL import Image



def saveims(images,batch,args,gstep,folder=None):
    prompts = batch['text']
    metas = batch['meta']
    for i in range(len(prompts)):
        if not args.eval_only and folder==None:
            name = prompts[i].replace(' ','_')[:30]
            images[i].save(osp.join(args.env.o,'vis',f'it-{gstep}-{name}.jpg'))
        else:
            folder_name = 'test' if folder is None else folder
            id1, id2 = metas[i]['id1'], metas[i]['id2']
            impath = osp.join(args.env.o,folder_name,f'{id1}_{id2}')
            os.makedirs(impath,exist_ok=True)
            images[i].save(osp.join(impath,f'eval-it-{gstep}.png'))
            with open(osp.join(impath,'prompt.txt'),'w') as f:
                f.write(prompts[i])


@torch.no_grad()
def val_fn(
        accelerator,
        args,
        vae,
        transformer,
        val_dataloader,
        weight_dtype,
        gstep,
        ):
    transformer.eval()
    pipe = DummyInferImage(
                           transformer.module if hasattr(transformer,'module') else transformer,
                           vae,args)
    for idx, batch in enumerate(val_dataloader):
        images = pipe(batch['text'], guidance_scale=args.eval.cfg).images
        saveims(images,batch,args,gstep)

    [setattr(blk.attn, "cache_kv", None) for blk in transformer.video_encoder.blocks]



class DummyInferImage:
    def __init__(self,
                 transformer,
                 vae,
                 args
                 ):
        self.transformer = transformer
        self.vae = vae
        self.args = args


    def __call__(self, *args, **kwds):
        return self.val_image(*args, **kwds)    
    
    @torch.no_grad()
    def val_image(
        self,
        prompt=None,
        num_inference_steps=64,
        num_diffusion_steps=25,
        max_latent_length=1,
        guidance_scale=5,
        motion_flow=5,
        negative_prompt=None,
        image=None,
        num_images_per_prompt=1,
        generator=None,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        output_type="pil",
        **kwargs,
    ) -> NOVAPipelineOutput:

        self.guidance_scale = guidance_scale
        inputs = {"generator": generator, **locals()}
        num_patches = int(np.prod(self.transformer.config.image_base_size))
        mask_ratios = np.cos(0.5 * np.pi * np.arange(num_inference_steps + 1) / num_inference_steps)
        mask_length = np.round(mask_ratios * num_patches).astype("int64")
        inputs["num_preds"] = mask_length[:-1] - mask_length[1:]    
        inputs["tqdm1"], inputs["tqdm2"] = max_latent_length > 1, max_latent_length == 1
        inputs["prompt"] = self.encode_prompt(**dict(_ for _ in inputs.items() if "prompt" in _[0]))
        inputs["latents"] = []
        inputs["batch_size"] = len(inputs["prompt"]) // (2 if guidance_scale > 1 else 1)
        inputs["motion_flow"] = [motion_flow] * inputs["batch_size"]
        _, outputs = inputs.pop("self"), self.transformer(inputs)
        outputs['x'] = outputs['x'].float()
        if self.args.zpad>0:
            _x = outputs['x']
            zpad = self.args.zpad
            assert _x.ndim==5
            _h = _x.shape[-2]
            outputs['x'] = torch.concat([_x[:,:,:,:,-zpad:],_x,_x[:,:,:,:,:zpad]],dim=-1)
        self.transformer.postprocess(outputs, {"vae": self.vae, **inputs})
        
        if self.args.zpad>0:
            _x = outputs['x']
            _,_nh,_,_c = _x.shape
            _ppad = _nh//_h*zpad
            assert _x.ndim==4 and _c == 3
            outputs['x'] = _x[:,:,_ppad:-_ppad,:]

        if output_type in ("latent", "pt"):
            return outputs["x"]
        outputs["x"] = outputs["x"].cpu().numpy()
        output_name = {4: "images", 5: "frames"}[len(outputs["x"].shape)]
        if output_type == "pil" and output_name == "images":
            outputs["x"] = [Image.fromarray(image) for image in outputs["x"]]
        return NOVAPipelineOutput(**{output_name: outputs["x"]})


    def encode_prompt(
        self,
        prompt,
        num_images_per_prompt=1,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ) -> torch.Tensor:

        def select_or_pad(a, b, n=1):
            return [a or b] * n if isinstance(a or b, str) else (a or b)

        embedder = self.transformer.text_embed
        if prompt_embeds is not None:
            prompt_embeds = embedder.encode_prompts(prompt_embeds)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = embedder.encode_prompts(negative_prompt_embeds)
        if prompt_embeds is not None:
            if negative_prompt_embeds is None and self.guidance_scale > 1:
                bs, seqlen = prompt_embeds.shape[:2]
                negative_prompt_embeds = embedder.weight[:seqlen].expand(bs, -1, -1)
            if self.guidance_scale > 1:
                c = torch.cat([prompt_embeds, negative_prompt_embeds])
            return c.repeat_interleave(num_images_per_prompt, dim=0)
        prompt = [prompt] if isinstance(prompt, str) else prompt
        negative_prompt = select_or_pad(negative_prompt, "", len(prompt))
        prompts = prompt + (negative_prompt if self.guidance_scale > 1 else [])
        c = embedder.encode_prompts(prompts)
        return c.repeat_interleave(num_images_per_prompt, dim=0)

