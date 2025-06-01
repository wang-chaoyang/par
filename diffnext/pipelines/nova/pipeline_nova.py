# Copyright (c) 2024-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
# Modified in 2025 by Chaoyang Wang

from typing import List

import numpy as np
import PIL.Image
import torch

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffnext.pipelines.nova.pipeline_utils import NOVAPipelineOutput, PipelineMixin


class NOVAPipeline(DiffusionPipeline, PipelineMixin):
    """NOVA autoregressive diffusion pipeline."""

    _optional_components = ["transformer", "scheduler", "vae", "text_encoder", "tokenizer"]
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    def __init__(
        self,
        transformer=None,
        scheduler=None,
        vae=None,
        text_encoder=None,
        tokenizer=None,
        trust_remote_code=True,
    ):
        super(NOVAPipeline, self).__init__()
        self.vae = self.register_module(vae, "vae")
        self.text_encoder = self.register_module(text_encoder, "text_encoder")
        self.tokenizer = self.register_module(tokenizer, "tokenizer")
        self.transformer = self.register_module(transformer, "transformer")
        self.scheduler = self.register_module(scheduler, "scheduler")
        self.transformer.sample_scheduler, self.guidance_scale = self.scheduler, 5.0
        self.tokenizer_max_length = self.transformer.text_embed.num_tokens
        self.transformer.text_embed.encoders = [self.tokenizer, self.text_encoder]

    @torch.no_grad()
    def __call__(
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
        inputs["latents"] = self.prepare_latents(image, num_images_per_prompt, generator, latents)
        inputs["batch_size"] = len(inputs["prompt"]) // (2 if guidance_scale > 1 else 1)
        inputs["motion_flow"] = [motion_flow] * inputs["batch_size"]
        _, outputs = inputs.pop("self"), self.transformer(inputs)
        self.transformer.postprocess(outputs, {"vae": self.vae, **inputs})
        if output_type in ("latent", "pt"):
            return outputs["x"]
        outputs["x"] = outputs["x"].cpu().numpy()
        output_name = {4: "images", 5: "frames"}[len(outputs["x"].shape)]
        if output_type == "pil" and output_name == "images":
            outputs["x"] = [PIL.Image.fromarray(image) for image in outputs["x"]]
        return NOVAPipelineOutput(**{output_name: outputs["x"]})

    def prepare_latents(
        self,
        image=None,
        num_images_per_prompt=1,
        generator=None,
        latents=None,
    ) -> List[torch.Tensor]:

        if latents is not None:
            return latents
        latents = []
        if image is not None:
            latents.append(self.encode_image(image, num_images_per_prompt, generator))
        return latents

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

    def encode_image(self, image, num_images_per_prompt=1, generator=None) -> torch.Tensor:

        x = torch.as_tensor(image, device=self.device).to(dtype=self.dtype)
        x = x.sub(127.5).div_(127.5).permute(2, 0, 1).unsqueeze_(0)
        x = self.vae.scale_(self.vae.encode(x).latent_dist.sample(generator))
        return x.expand(num_images_per_prompt, -1, -1, -1)
