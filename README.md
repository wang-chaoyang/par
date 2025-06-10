# Conditional Panoramic Image Generation via Masked Autoregressive Modeling


<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2505.16862-b31b1b.svg)](https://arxiv.org/abs/2505.16862)
[![Project Website](https://img.shields.io/badge/🔗-Project_Website-blue.svg)](https://wang-chaoyang.github.io/project/par)

</div>

<div>
  <p align="center" style="font-size: larger;">
    <strong>Arxiv</strong>
  </p>
</div>

<p align="center">
<img src="https://wang-chaoyang.github.io/project/par/static/images/teaser.png" width=95%>
<p>

## Requirements


Install other pip packages via `pip install -r requirements.txt`.


## Preparation

Download the [pre-trained model](https://github.com/baaivision/NOVA/blob/main/docs/model_zoo.md) and put them under `data/pretrain`.

We use [Matterport3D](https://niessner.github.io/Matterport) in our experiments and follow [PanFusion](https://github.com/chengzhag/PanFusion) for data preparation. The data should be organized as follows.
```
data
├── Matterport3D
│   ├── mp3d_skybox
│   │   ├── 1LXtFkjw3qL
│   │       ├── matterport_skybox_images
│   │       └── matterport_stitched_images   
│   │   ├── train.npy   
│   │   └── test.npy  
│   └── caption
│       ├── 1LXtFkjw3qL_0b22fa63d0f54a529c525afbf2e8bb25.txt
└── pretrain
```

## Demo

Download the [checkpoint](https://huggingface.co/chaoyangw/par/blob/main/par_w1024.pt) and put it under `data/pretrain/ckpt/par_w1024.pt`.


```
accelerate launch scripts/demo.py --ptr data/pretrain/nova-d48w1024-sdxl1024 --ckpt data/pretrain/ckpt/par_w1024.pt --prompt=[Prompt]
```

## Train
The codes in [scripts](scripts/train.py) is launched by accelerate. The images and captions are specified by `path` and `textpath`, respectively.
```
accelerate launch scripts/train.py configs/cfg.yaml ptr=[ptr] path=[path] textpath=[textpath] env.o=[project path]
```

## Inference
```
accelerate launch scripts/train.py configs/cfg.yaml ptr=[ptr] path=[path] textpath=[textpath] env.o=[project path] eval_only=true 
```


## Citation

If you find our work interesting, please kindly consider citing our paper:

```bibtex
@article{wang2025conditional,
    title={Conditional Panoramic Image Generation via Masked Autoregressive Modeling},
    author={Wang, Chaoyang and Li, Xiangtai and Qi, Lu and Lin, Xiaofan and Bai, Jinbin and Zhou, Qianyu and Tong, Yunhai},
    journal={arXiv preprint arXiv:2505.16862},
    year={2025}
}
```

## License

Apache License 2.0