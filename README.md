# PatchAD

This is the implementation of *PatchAD: A Lightweight Patch-based MLP-Mixer for Time Series Anomaly Detection*.

You can download the paper from [arXiv](https://arxiv.org/abs/2401.09793).

# Abstract

Anomaly detection in time series analysis is a pivotal task, yet it poses the challenge of discerning normal and abnormal patterns in label-deficient scenarios. While prior studies have largely employed reconstruction-based approaches, which limits the models' representational capacities. Moreover, existing deep learning-based methods are often not sufficiently lightweight. Addressing these issues, we present PatchAD, our novel, highly efficient multi-scale patch-based MLP-Mixer architecture that utilizes contrastive learning for representation extraction and anomaly detection. With its four distinct MLP Mixers and innovative dual project constraint module, PatchAD mitigates potential model degradation and offers a lightweight solution, requiring only **3.2MB**. Its efficacy is demonstrated by state-of-the-art results across **nine** datasets, outperforming over **30** comparative algorithms. PatchAD significantly improves the classical F1 score by **50.5\%**, the Aff-F1 score by **7.8\%**, and the AUC by **10.0\%**. The code is publicly available.

## Architecture

![Architecture](./paper_img/fw2.png)

## Overall Performance \& Model Size

<div style="display: flex; flex-wrap: nowrap;">
  <img src="./paper_img/pfrm.png" alt="Overall Performance" style="max-width: 300px; margin-right: 10px;" width=450px;>
  <img src="./paper_img/mdlsz.png" alt="Model Size" style="max-width: 150px;" width=200px; >
</div>

## Datasets

You can download all datasets [here](https://drive.google.com/drive/folders/1RaIJQ8esoWuhyphhmMaH-VCDh-WIluRR?usp=sharing).
(Thanks for [DCdetector](https://github.com/DAMO-DI-ML/KDD2023-DCdetector/blob/main/readme.md) repo and its authors.)

## Installation

You can refer to requirements.txt to install all the packages.

[^2]: **We have not tested it to make sure it can be installed successfully. We will test it in the future.**
    
> pip install -r requirements.txt

## Quick start

1. You should download the datasets into ABSOLUTE/PATH/OF/DATASET.
2. The dataset structure should be like *'dataset_struct.txt'*
3. Run the scripts below.
4. Note that the most important is you should change the parameter of *--data_path*.
5. We use *--model_save_path* and *--res_pth* for model and result saving.
6. TODO

Train

> python main_ad.py --anormly_ratio 0.9 -ep 3  --data_path ABSOLUTE/PATH/OF/DATASET --batch_size 128  --mode train --data_name PSM --win_size 105 --stride 1 --patch_size [3,5,7] --patch_mx 0.1 --d_model 60 --e_layer 3 -lr 0.0001

Test

> python main_ad.py --anormly_ratio 0.9 -ep 3  --data_path ABSOLUTE/PATH/OF/DATASET --batch_size 128  --mode test --data_name PSM --win_size 105 --stride 1 --patch_size [3,5,7] --patch_mx 0.1 --d_model 60 --e_layer 3 -lr 0.0001

## Citation

**If you find this repo useful, please cite our paper.**

> @misc{zhong2024patchad, `<br>`
> &nbsp;&nbsp;&nbsp;&nbsp;title={PatchAD: A Lightweight Patch-based MLP-Mixer for Time Series Anomaly Detection}, `<br>`
> &nbsp;&nbsp;&nbsp;&nbsp;author={anonymous authors}, `<br>`
> &nbsp;&nbsp;&nbsp;&nbsp;year={2024}, `<br>`
> &nbsp;&nbsp;&nbsp;&nbsp;eprint={2401.09793}, `<br>`
> &nbsp;&nbsp;&nbsp;&nbsp;archivePrefix={arXiv}, `<br>`
> &nbsp;&nbsp;&nbsp;&nbsp;primaryClass={cs.LG} `<br>`
> }
