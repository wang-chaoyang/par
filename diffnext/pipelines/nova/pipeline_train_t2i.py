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

from typing import Dict

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
import torch

from diffnext.pipelines.builder import PIPELINES, build_diffusion_scheduler
from diffnext.pipelines.nova.pipeline_utils import PipelineMixin, freeze_module

from diffnext.models.transformers.transformer_nova import NOVATransformer3DModel
from diffnext.models.text_encoders.phi import PhiEncoderModel
from diffnext.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffnext.schedulers.scheduling_flow import FlowMatchEulerDiscreteScheduler


@PIPELINES.register("nova_train_t2i")
class NOVATrainT2IPipeline(DiffusionPipeline, PipelineMixin):
    """Pipeline for training NOVA models."""

    _optional_components = ["transformer", "scheduler", "vae", "text_encoder", "tokenizer"]

    def __init__(
        self,
        transformer:NOVATransformer3DModel=None,
        scheduler:FlowMatchEulerDiscreteScheduler=None,
        vae:AutoencoderKL=None,
        text_encoder:PhiEncoderModel=None,
        tokenizer=None,
        trust_remote_code=True,
    ):
        super(NOVATrainT2IPipeline, self).__init__()
        self.vae = self.register_module(vae, "vae")
        self.text_encoder = self.register_module(text_encoder, "text_encoder")
        self.tokenizer = self.register_module(tokenizer, "tokenizer")
        self.transformer = self.register_module(transformer, "transformer")
        self.scheduler = self.register_module(scheduler, "scheduler")
        self.transformer.noise_scheduler = build_diffusion_scheduler(self.scheduler)
        self.transformer.sample_scheduler, self.guidance_scale = self.scheduler, 5.0

    @property
    def model(self) -> torch.nn.Module:
        """Return the trainable model."""
        return self.transformer

    def configure_model(self, loss_repeat=4, checkpointing_level=0, config=None) -> torch.nn.Module:
        """Configure the trainable model."""
        self.model.loss_repeat = config.TRAIN.LOSS_REPEAT if config else loss_repeat
        ckpt_lvl = config.TRAIN.CHECKPOINTING_LEVEL if config else checkpointing_level
        [setattr(blk, "mlp_checkpointing", ckpt_lvl) for blk in self.model.video_encoder.blocks]
        [setattr(blk, "mlp_checkpointing", ckpt_lvl > 1) for blk in self.model.image_encoder.blocks]
        [setattr(blk, "mlp_checkpointing", ckpt_lvl > 2) for blk in self.model.image_decoder.blocks]
        freeze_module(self.model.text_embed.norm)  # We always use frozen LN.
        freeze_module(self.model.video_pos_embed)  # Freeze this module during T2I.
        freeze_module(self.model.video_encoder.patch_embed)  # Freeze this module during T2I.
        freeze_module(self.model.motion_embed) if self.model.motion_embed else None
        self.model.pipeline_preprocess = self.preprocess
        self.model.text_embed.encoders = [self.tokenizer, self.text_encoder]
        return self.model.train()

    def prepare_latents(self, inputs: Dict):
        """Prepare the video latents."""
        if "images" in inputs:
            raise NotImplementedError
        elif "moments" in inputs:
            x = torch.as_tensor(inputs.pop("moments"), device=self.device).to(dtype=self.dtype)
            inputs["x"] = self.vae.scale_(self.vae.latent_dist(x).sample())

    def encode_prompt(self, inputs: Dict):
        """Encode text prompts."""
        inputs["c"] = inputs.get("c", [])
        if inputs.get("prompt", None) is not None and self.transformer.text_embed:
            inputs["c"].append(self.transformer.text_embed(inputs.pop("prompt")))

    def preprocess(self, inputs: Dict) -> Dict:
        """Define the pipeline preprocess at every call."""
        if not self.model.training:
            raise RuntimeError("Excepted a trainable model.")
        self.prepare_latents(inputs)
        self.encode_prompt(inputs)
