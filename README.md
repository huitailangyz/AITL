# Adaptive Image Transformations for Transfer-based Adversarial Attack

This repository is an official implementation of the paper [Adaptive Image Transformations for Transfer-based Adversarial Attack](https://arxiv.org/abs/2111.13844), which has been accepted by ECCV2022.



## Abstract

Adversarial attacks provide a good way to study the robustness of deep learning models. One category of methods in transfer-based black-box attack utilizes several image transformation operations to improve the transferability of adversarial examples, which is effective, but fails to take the specific characteristic of the input image into consideration. In this work, we propose a novel architecture, called Adaptive Image Transformation Learner (AITL), which incorporates different image transformation operations into a unified framework to further improve the transferability of adversarial examples. Unlike the fixed combinational transformations used in existing works, our elaborately designed transformation learner adaptively selects the most effective combination of image transformations specific to the input image. Extensive experiments on ImageNet demonstrate that our method significantly improves the attack success rates on both normally trained models and defense models under various settings.



## Environment
 - python==`3.6.5`
 - tensorflow==`1.14`
 - numpy==`1.19.4`
 - imageio==`2.3.0`
 - pandas=`1.1.5`
 - stn==`1.0.1`


## Dataset
 -  [Training set](https://github.com/cleverhans-lab/cleverhans/tree/master/cleverhans_v3.1.0/examples/nips17_adversarial_competition/dataset) Download the images and the label file to `data/dev_images` and `data/dev_dataset.csv` respectively
  
 -   [Evaluation set](https://drive.google.com/drive/folders/1CfobY6i8BfqfWPHL31FKFDipNjqWwAhS) Download the images and the label file to `data/val_rs` and `data/val_rs.csv` respectively

## Model
 - [Naturally trained models](https://github.com/tensorflow/models/tree/master/research/slim): Incv3, Incv4, IncResv2, Resv2-101, Resv2-152, Mobv2-1.0, Mobv2-1.4, PNASNet, NASNet
 - [Adversarially trained models](https://github.com/wowowoxuan/adv_imagenet_models): Incv3_adv, Incv3_ens3, Incv3_ens4, IncResv2_ens 
  
    Download all the models to `model`
## Script
### training AITL
    bash train.sh gpu-id
The trained model will be saved in `model/AITL_20_4`

### attack with pre-trained AITL
    bash attack.sh gpu-id
The generated adversarial images and corresponding evaluation result will be saved in `output`

We've has provided an [example](https://drive.google.com/drive/folders/1-Ngh0FjVVkuj2Jbrb9eLt8pNEiBcLlUH?usp=sharing) of the generated adversarial images (e.g., 'AITL (ours)' in Table 3 in the paper) for evaluation.