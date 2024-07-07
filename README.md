

<div align="center">

<samp>

<h2> EndoUIC: Promptable Diffusion Transformer for Unified Illumination Correction in Capsule Endoscopy </h1>

<h4> Long Bai*, Qiaozhi Tan*, Tong Chen*, Wanjun Nah, Yanheng Li, Zhicheng He, Sishen Yuan, Zhen Chen, Jinlin Wu, Mobarakol Islam, Zhen Li, Hongbin Liu, and Hongliang Ren </h3>

<h3> Medical Image Computing and Computer Assisted Intervention (MICCAI) 2024 </h2>

</samp>   

| **[[```arXiv```](<https://arxiv.org/abs/2307.02452>)]** | **[[```Paper```](<https://link.springer.com/chapter/10.1007/978-3-031-43999-5_4>)]** |
|:-------------------:|:-------------------:|

---

</div>     

If you find our code, paper, or dataset useful, please cite the paper as

```bibtex
@inproceedings{bai2023llcaps,
  title={LLCaps: Learning to Illuminate Low-Light Capsule Endoscopy with Curved Wavelet Attention and Reverse Diffusion},
  author={Bai, Long and Chen, Tong and Wu, Yanan and Wang, An and Islam, Mobarakol and Ren, Hongliang},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={34--44},
  year={2023},
  organization={Springer}
}
```
--- 

## Abstract

Wireless Capsule Endoscopy (WCE) is highly valued for its non-invasive and painless approach, though its effectiveness is compromised by uneven illumination from hardware constraints and complex internal dynamics, leading to overexposed or underexposed images. While researchers have discussed the challenges of low-light enhancement in WCE, the issue of correcting for different exposure levels remains underexplored. To tackle this, we introduce EndoUIC, a WCE unified illumination correction solution using an end-to-end promptable diffusion transformer (DFT) model. In our work, the illumination prompt module shall navigate the model to adapt to different exposure levels and perform targeted image enhancement, in which the Adaptive Prompt Integration (API) and Global Prompt Scanner (GPS) modules shall further boost the concurrent representation learning between the prompt parameters and features. Besides, the U-shaped restoration DFT model shall capture the long-range dependencies and contextual information for unified illumination restoration. Moreover, we present a novel Capsule-endoscopy Exposure Correction (CEC) dataset, including ground-truth and corrupted image pairs annotated by expert photographers. Extensive experiments against a variety of state-of-the-art (SOTA) methods on four datasets showcase the effectiveness of our proposed method and components in WCE illumination restoration, and the additional downstream experiments further demonstrate its utility for clinical diagnosis and surgical assistance. 


---
## Environment

For environment setup, please follow these instructions
```
git clone https://github.com/longbai-cuhk/EndoUIC.git
cd EndoUIC
conda create -n EndoUIC python=3.7
conda activate EndoUIC
conda install pytorch==1.7.0 torchvision torchaudio cudatoolkit=11.0 -c pytorch
cd BasicSR-light
pip install -r requirements.txt
BASICSR_EXT=True sudo $(which python) setup.py develop
cd ../EndoUIC
pip install -r requirements.txt
BASICSR_EXT=True sudo $(which python) setup.py develop
```

---
## Dataset
1. [Kvasir-Capsule Dataset](https://osf.io/dv2ag/) and [Red Lesion Endoscopy Dataset](https://rdm.inesctec.pt/dataset/nis-2018-003)
    - The low-light and ground-truth image pairs are released by [LLCaps](https://github.com/longbai1006/LLCaps). 
2. [Endo4IE Dataset](https://data.mendeley.com/datasets/3j3tmghw33/1)
3. [Capsule endoscopy Exposure Correction (CEC) Dataset](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155161502_link_cuhk_edu_hk/EZuLCQk1SjRMr7L6pIpiG5kBwhcMGp1hB_g73lySKlVUjA?e=g84Zl8)
---


## Training

Train your model with one GPU by running

```
CUDA_VISIBLE_DEVICES=0 python endouic/train.py -opt options/train_v1.yaml --launcher pytorch
```
Training arguments can be modified in 'train_v1.yml'.

Train your model with two or more GPUs by running

```
CUDA_VISIBLE_DEVICES=0,1...,n-1 python -m torch.distributed.launch --nproc_per_node=n --master_port=12345 endouic/train.py -opt options/train_v1.yaml --launcher pytorch
```
Training arguments can be modified in 'train_v1.yml'.

## Inference
Conduct model inference by running

```
CUDA_VISIBLE_DEVICES=0 python endouic/train.py -opt options/infer.yaml
```

## Evaluation (PSNR, SSIM, LPIPS, ....)

Please install the dependency needed.

```
cd evaluation
python evaluation.py -dir_A /[GT_PATH] -dir_B /[GENERATED_IMAGE_PATH] 
```

---


