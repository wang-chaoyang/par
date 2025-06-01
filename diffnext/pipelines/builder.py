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

import torch

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffnext.utils.registry import Registry

PIPELINES = Registry("pipelines")


def build_diffusion_scheduler(scheduler_path, sample=False, **kwargs) -> SchedulerMixin:
    """Create a diffusion scheduler instance.

    Args:
        scheduler_path (str or scheduler instance)
            The path to load a diffusion scheduler.
        sample (bool, *optional*, default to False)
            Whether to create the sampling-specific scheduler.

    """
    from diffnext.schedulers.scheduling_ddpm import DDPMScheduler
    from diffnext.schedulers.scheduling_flow import FlowMatchEulerDiscreteScheduler  # noqa

    if isinstance(scheduler_path, str):
        class_key = "_{}_class_name".format("sample" if sample else "noise")
        class_type = locals()[DDPMScheduler.load_config(**locals())[class_key]]
        return class_type.from_pretrained(**locals())
    elif hasattr(scheduler_path, "config"):
        class_type = locals()[type(scheduler_path).__name__]
        return class_type.from_config(scheduler_path.config)
    return None


def build_pipeline(
    path=None,
    pipe_type=None,
    precison="bfloat16",
    config=None,
    **kwargs,
) -> DiffusionPipeline:
    """Create a diffnext pipeline instance.

    Examples:
        ```py
        >>> from diffnext.pipelines import build_pipeline
        >>> pipe = build_pipeline("BAAI/nova-d48w768-sdxl1024", "nova_train_t2i")
        ```

    Args:
        path (str, *optional*):
            The model path that includes ``model_index.json`` to create pipeline.
        pipe_type (str, *optional*)
            The registered pipeline class.
        precision (str, *optional*, default to ``bfloat16``)
            The compute precision used for all pipeline components.
        cfg (object, *optional*)
            The config object.

    """
    path = config.MODEL.PIPELINE_PATH if config else path
    pipe_type = config.MODEL.PIPELINE_TYPE if config else pipe_type
    precison = config.MODEL.PRECISION if config else precison
    kwargs.setdefault("trust_remote_code", True)
    kwargs.setdefault("torch_dtype", getattr(torch, precison.lower()))
    return PIPELINES.get(pipe_type).func.from_pretrained(path, **kwargs)
