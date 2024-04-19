# 6Img-to-3D

### [Project Page](https://6img-to-3d.github.io/) | [Videos](https://www.youtube.com/@6Img-to-3D) | [Paper](https://arxiv.org/abs/2404.12378) | [Data](Published_Soon)

[6Img-to-3D: Few-Image Large-Scale Outdoor Driving Scene Reconstruction](https://6img-to-3d.github.io/6img-to-3D/)  
 [Theo Gieruc](https://github.com/tgieruc)\*<sup>1</sup><sup>2</sup>, [Marius Kaestingschaefer](https://marius.cx/)\*<sup>1</sup>, [Sebastian Bernhard](https://www.linkedin.com/in/dr-ing-sebastian-bernhard-79a763205/)<sup>1</sup>, [Mathieu Salzmann](https://people.epfl.ch/mathieu.salzmann)<sup>2</sup>,
 
 <sup>1</sup>Continental AG, <sup>2</sup>EPFL
  \*denotes equal contribution  

A PyTorch implementation of the 6Img-to-3D model for large-scale outdoor driving scene reconstruction. The model takes as input six images from a driving scene and outputs a parameterized triplane from which novel views can be rendered.

<p align="center">
  <img src="media\driving.gif" alt="Driving" style="width: 120%;" />
</p>

If you find this code useful, please reference in your paper:

```
@misc{gieruc20246imgto3d,
      title={6Img-to-3D: Few-Image Large-Scale Outdoor Driving Scene Reconstruction}, 
      author={Théo Gieruc and Marius Kästingschäfer and Sebastian Bernhard and Mathieu Salzmann},
      year={2024},
      eprint={2404.12378},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
## 6Img-to-3D

Inward and outward-facing camera setups differ significantly in their view overlap. Outward-facing (inside-out) camera setups overlap minimally, whereas inward-facing (outside-in) setups can overlap across multiple cameras.

<p align="center">
  <img src="media\views.png" alt="Views" style="width: 50%;" />
</p>

Given six input images, we first encode them into feature maps using a pre-trained ResNet and an FPN. The scene coordinates are contracted to fit the unbounded scenes. MLPs, cross-and self-attention layers form the Image-to-Triplane Encoder of our framework. Images can be rendered from the resulting triplane using our renderer. We additionally condition the rendering process on projected image features.

<p align="center">
  <img src="media\method.png" alt="Method" style="width: 100%;" />
</p>


## Installation
```bash
conda create -n sixtothree python=3.8
conda activate sixtothree
```

Install PyTorch 2.0.1 with CUDA 11.8 (recommanded), cuda-toolkit and tinycudann.
```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

Install the MMLAB suite
```bash
pip install mmdet==2.20.0 mmengine==0.8.4 mmsegmentation==0.20.0  mmcls==0.25.0 mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.1/index.html
```

Install the other dependencies
```bash
pip install tensorboardX crc32c pandas pyyaml==6.0.1  imageio==2.33.1 imageio-ffmpeg==0.4.9 lpips==0.1.4 pytorch-msssim==1.0.0 kornia==0.7.0 yapf==0.40.1 jupyter notebook seaborn==0.13.2
```

## Train
To train the model use the [train.py](train.py) script

### Pre-pickle train dataset
So that the training runs faster, we first turn the train dataset into pickles with the [pickles_generator.py](utils/pickles_generator.py) script.

```bash
 python utils/generator_pickles.py --dataset-config config/_base_/dataset.py --py-config config/config.py
```

### Usage

To train the model, run the training script with the desired arguments specified using the command line interface. Here's how to use each argument:

- `--py-config`: Path to the Python configuration file (.py) containing model configurations. This file specifies the architecture and parameters of the model being trained.

- `--ckpt-path`: Path to a TPVFormer checkpoint file to initialize model weights from. If specified, the training will resume from this checkpoint.

- `--resume-from`: Path to a checkpoint file from which to resume training. This option allows you to continue training from a specific checkpoint.

- `--log-dir`: Directory where Tensorboard training logs and saved models will be stored. If not provided, logs will be saved in the default directory with a timestamp.

- `--num-scenes`: Specifies the number of scenes to train on. This argument allows for faster training when only a subset of scenes is required.

- `--from-epoch`: Specifies the starting epoch for training. If training is interrupted and resumed, you can specify the epoch from which to resume training.


### Running the Script

To run the train9ing script, execute the Python file `train.py` with the desired arguments specified using the command line interface. For example:

```bash
python train.py --py-config config/config.py --ckpt-path ckpts/tpvformer.pth --log-dir evaluation_results 
```


## Eval
To evaluate the model, use the [eval.py](eval.py) script.

### Usage

The evaluation script can be run with different options to customize the evaluation process. Here's how to use each argument:

- `--py-config`: Path to the Python configuration file (.py) containing model configurations. This file specifies the architecture and parameters of the model being evaluated.

- `--dataset-config`: Path to the dataset configuration file (.py) containing dataset parameters. This file specifies dataset-specific settings such as image paths and scalling.

- `--resume-from`: Path to the checkpoint file from which to resume model evaluation. This argument allows you to continue evaluation from a previously saved checkpoint.

- `--log-dir`: Directory where evaluation Tensorboard logs and results will be saved. The default behavior is to create a directory with a timestamp indicating the evaluation start time.

- `--depth`: If specified, depth maps will also be saved.

- `--gif`: If specified, the script generates GIFs from the evaluated images.

- `--gif-gt`: If specified, GIFs are generated for ground truth images.

- `--img-gt`: If specified, the script saves ground truth images alongside the generated images. 

- `--num-img`: Specifies the number of images to evaluate. By default, all images in the dataset are evaluated. This argument allows for faster evaluation when only a subset of images is required.

- `--time`: Compute inference time of the model and save results in `t_decode.txt`, `t_encode.txt`



### Running the Script

To run the evaluation script, execute the Python file `eval.py` with the desired arguments specified using the command line interface. For example:

```bash
python eval.py --py-config ckpts/6Img-to-3D/config.py --resume-from ckpts/6Img-to-3D/model_checkpoint.pth --log-dir evaluation_results --depth --img-gt --dataset-config config/_base_/dataset_eval.py
```

### License
Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG). All rights reserved.
This repository is licensed under the BSD-3-Clause license. See [LICENSE](./LICENSE) for the full license text.
