# MedSAM
This is the official repository for MedSAM: Segment Anything in Medical Images.

## News

- 2024.01.15: Welcome to join [CVPR 2024 Challenge: MedSAM on Laptop](https://www.codabench.org/competitions/1847/)
- 2024.01.15: Release [LiteMedSAM](https://github.com/bowang-lab/MedSAM/blob/LiteMedSAM/README.md) and [3D Slicer Plugin](https://github.com/bowang-lab/MedSAMSlicer), 10x faster than MedSAM! 


## Installation
1. Create a virtual environment `conda create -n medsam python=3.10 -y` and activate it `conda activate medsam`
2. Install `pip3 install torch torchvision torchaudio`
3. `git clone https://github.com/DLYuanGod/BioSAM2`
4. Enter the BioSAM2 folder `cd BioSAM2` and run `pip install -e .`
5. Enter `cd segment-anything-2` and `pip install -e . ; cd ..`






## Model Training

### Data preprocessing

Download [SAM2 checkpoint] `cd segment-anything-2/checkpoints;./download_ckpts.sh` and place it at [here](https://github.com/DLYuanGod/BioSAM2/blob/main/train_one_gpu.py#279) and change the cfg [here](https://github.com/DLYuanGod/BioSAM2/blob/main/train_one_gpu.py#280).

Download the [dataset](https://drive.google.com/drive/folders/18QSSiABS8H3qtx8SZA6RQb3aH1nbc3iF).

Run pre-processing

#### Datasets 701 and 702:

Install `cc3d`: `pip install connected-components-3d` and change the path [here](https://github.com/DLYuanGod/BioSAM2/blob/main/pre_CT_MR.py#22) and [here](https://github.com/DLYuanGod/BioSAM2/blob/main/pre_CT_MR.py#23) and output dir [here](https://github.com/DLYuanGod/BioSAM2/blob/main/pre_CT_MR.py#24).

```bash
python pre_CT_MR.py
```

#### Datasets 703 and 704

Change the path [here](https://github.com/DLYuanGod/BioSAM2/blob/main/pre_CELL_EN.py#14) and [here](https://github.com/DLYuanGod/BioSAM2/blob/main/pre_CELL_EN.py#15) and output dir [here](https://github.com/DLYuanGod/BioSAM2/blob/main/pre_CELL_EN.py#16).

```bash
python pre_CELL_EN.py
```


### Training on one GPU

Change the data npy path [here](https://github.com/DLYuanGod/BioSAM2/blob/main/train_one_gpu.py#122) and [here](https://github.com/DLYuanGod/BioSAM2/blob/main/train_one_gpu.py#154).

```bash
python train_one_gpu.py
```



