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



## Demo

Download the [checkpoint](https://huggingface.co/chaoyangw/par/blob/main/par_w1024.pt) and put it under `data/pretrain/ckpt/par_w1024.pt`.


```
accelerate launch scripts/demo.py --ptr data/pretrain/nova-d48w1024-sdxl1024 --ckpt data/pretrain/ckpt/par_w1024.pt --prompt=[Prompt]
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