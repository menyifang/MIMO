# MIMO - Official PyTorch Implementation

### [Project page](https://menyifang.github.io/projects/MIMO/index.html) | [Paper](https://arxiv.org/abs/2409.16160) | [Video](https://www.youtube.com/watch?v=skw9lPKFfcE) | [Online Demo](https://modelscope.cn/studios/iic/MIMO)

> **MIMO: Controllable Character Video Synthesis with Spatial Decomposed Modeling**<br>
> [Yifang Men](https://menyifang.github.io/), [Yuan Yao](mailto:yaoy92@gmail.com), [Miaomiao Cui](mailto:miaomiao.cmm@alibaba-inc.com), [Liefeng Bo](https://scholar.google.com/citations?user=FJwtMf0AAAAJ&hl=en)<br>
> Institute for Intelligent Computing (Tongyi Lab), Alibaba Group
> In: CVPR 2025 

MIMO is a generalizable model for controllable video synthesis, which can not only synthesize realistic character videos with controllable attributes (i.e., character, motion and scene) provided by very simple user inputs, but also simultaneously achieve advanced scalability to arbitrary characters, generality to novel 3D motions, and applicability to interactive real-world scenes in a unified framework. 

## Demo

Animating character image with driving 3D pose from motion dataset

https://github.com/user-attachments/assets/3a13456f-9ee5-437c-aba4-30d8c3b6e251

Driven by in-the-wild video with spatial 3D motion and interactive scene

https://github.com/user-attachments/assets/4d989e7f-a623-4339-b3d1-1d1a33ad25f2


More results can be found in [project page](https://menyifang.github.io/projects/MIMO/index.html).


## 📢 News
(2025-06-11) The code is released! We released a simplified version of full implementation, but it could achieve comparable performance.

(2025-02-27) The paper is accepted by CVPR 2025! The full version of the paper is available on [arXiv](https://arxiv.org/abs/2409.16160).

(2024-01-07) The online demo (v1.5) supporting custom driving videos is available now! Try out [![ModelScope Spaces](
https://img.shields.io/badge/ModelScope-Spaces-blue)](https://modelscope.cn/studios/iic/MIMO).

(2024-11-26) The online demo (v1.0) is available on ModelScope now! Try out [![ModelScope Spaces](
https://img.shields.io/badge/ModelScope-Spaces-blue)](https://modelscope.cn/studios/iic/MIMO). The 1.5 version to support custom driving videos will be coming soon.

(2024-09-25) The project page, demo video and technical report are released. The full paper version with more details is in process.



## Requirements
* python (>=3.10)
* pyTorch
* tensorflow
* cuda 12.1
* GPU (tested on A100, L20)


## 🚀 Getting Started

```bash
git clone https://github.com/menyifang/MIMO.git
cd MIMO
```

### Installation
```bash
conda create -n mimo python=3.10
conda activate mimo
bash install.sh
```

### Downloads

#### Model Weights 

You can manually download model weights from [ModelScope](https://modelscope.cn/models/iic/MIMO/files) or [Huggingface](https://huggingface.co/menyifang/MIMO/tree/main), or automatically using follow commands.

Download from HuggingFace
```python
from huggingface_hub import snapshot_download 
model_dir = snapshot_download(repo_id='menyifang/MIMO', cache_dir='./pretrained_weights')
```

Download from ModelScope 
```python
from modelscope import snapshot_download
model_dir = snapshot_download(model_id='iic/MIMO', cache_dir='./pretrained_weights')
```


#### Prior Model Weights 

Download pretrained weights of based model and other components: 
- [StableDiffusion V1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
- [image_encoder](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/tree/main/image_encoder)


#### Data Preparation

Download examples and resources (`assets.zip`) from [google drive](https://drive.google.com/file/d/1qf4sSQggAJZUnBP0GLHkVR12IjeKw_6j/view?usp=drive_link) and unzip it under `${PROJECT_ROOT}/`.
You can also process custom videos following [Process driving templates](#process-driving-templates).

After downloading weights and data, the folder of the project structure seems like:

```text
./pretrained_weights/
|-- image_encoder
|   |-- config.json
|   `-- pytorch_model.bin
|-- denoising_unet.pth
|-- motion_module.pth
|-- pose_guider.pth
|-- reference_unet.pth
|-- sd-vae-ft-mse
|   |-- config.json
|   |-- diffusion_pytorch_model.bin
|   `-- diffusion_pytorch_model.safetensors
`-- stable-diffusion-v1-5
    |-- feature_extractor
    |   `-- preprocessor_config.json
    |-- model_index.json
    |-- unet
    |   |-- config.json
    |   `-- diffusion_pytorch_model.bin
    `-- v1-inference.yaml
./assets/
|-- video_template
|   |-- template1

```

Note: If you have installed some of the pretrained models, such as `StableDiffusion V1.5`, you can specify their paths in the config file (e.g. `./config/prompts/animation_edit.yaml`).


### Inference

- video character editing
```bash
python run_edit.py
```

- character image animation
```bash
python run_animate.py
```


### Process driving templates

- install external dependencies by
```bash
bash setup.sh
```
you can also use dockerfile(`video_decomp/docker/decomp.dockerfile`) to build a docker image with all dependencies installed.


- download model weights and data from [Huggingface](https://huggingface.co/menyifang/MIMO_VidDecomp/tree/main) and put them under `${PROJECT_ROOT}/video_decomp/`.

```python
from huggingface_hub import snapshot_download 
model_dir = snapshot_download(repo_id='menyifang/MIMO_VidDecomp', cache_dir='./video_decomp/')
```


- process the driving video by
```bash
cd video_decomp
python run.py
```

The processed template can be putted under `${PROJECT_ROOT}/assets/video_template` for editing and animation tasks as follows:
```
./assets/video_template/
|-- template1/
|   |-- vid.mp4
|   |-- mask.mp4
|   |-- sdc.mp4
|   |-- bk.mp4
|   |-- occ.mp4 (if existing)
|-- template2/
|-- ...
|-- templateN/
```

### Training



## 🎨 Gradio Demo

**Online Demo**: We launch an online demo of MIMO at [ModelScope Studio](https://modelscope.cn/studios/iic/MIMO).

If you have your own GPU resource (>= 40GB vram), you can run a local gradio app via following commands:

`python app.py`



## Acknowledgments

Thanks for great work from [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone), [SAM](https://github.com/facebookresearch/segment-anything), [4D-Humans](https://github.com/shubham-goel/4D-Humans), [ProPainter](https://github.com/sczhou/ProPainter)


## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@inproceedings{men2025mimo,
  title={MIMO: Controllable Character Video Synthesis with Spatial Decomposed Modeling},
  author={Men, Yifang and Yao, Yuan and Cui, Miaomiao and Liefeng Bo},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2025 IEEE Conference on},
  year={2025}}
}
```



