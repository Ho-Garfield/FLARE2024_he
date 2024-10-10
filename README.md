# Joint Unsupervised Domain Adaptation and Semi-Supervised Learning for Multi-Sequence MR Abdominal Organ Segmentation
[![flow](https://github.com/Ho-Garfield/-FLARE2024_solution_he/blob/main/1.png)](flow)


## Environments and Requirements

- Ubuntu 22.04.2 LTS
- Intel(R) Xeon(R) Silver 4210R CPU @ 2.40GHz, 8 × 32GB; 2400MT/s, NVIDIA GeForce RTX 4090 24G
- 11.8
- Python 3.9.0
  

### 1. install miniconda
https://docs.anaconda.com/miniconda/
### 2. install requirements:
```setup
cd miccai_model
conda create -n miccai python=3.9
conda activate miccai
pip install -e .
```



## Dataset

https://www.codabench.org/competitions/2296/

## Run window.py
```setup
python window.py
```
You can see that the window contains preprocessing, stage1, stage2, and stage3, as shown in the figure below.
[![flow](https://github.com/Ho-Garfield/-FLARE2024_solution_he/blob/main/2.png)](flow)

## Prepare the data
Create the ```miccai_model/FLARE24``` , and place the data in the following format.
```
FLARE24/
├── CT/
│   ├── images/
│   │   ├── ..._0000.nii.gz
│   │   └── ...
│   ├── labels/
│   │   ├── ...nii.gz
│   │   └── ...
├── MRI/
│   ├── AMOS/
│   │   ├── ..._0000.nii.gz
│   │   └── ...
│   ├── LLD/
│   │   ├── ..._0000.nii.gz
│   │   └── ...


```
## Preprocessing

Click preprocess to enter the preprocessing window.
[![flow](https://github.com/Ho-Garfield/-FLARE2024_solution_he/blob/main/3.jpg)](flow)


### 1. Resampling. 


### 2. Patient position readjustment.
   


### 3. Adjust grayscale range.

### 4. Translation registration.(CT<->MRI T1W）

## Stage 1
### 1. 3D CycleGAN
In this project, the implementation of the 3D CycleGAN is based on the repository available at https://github.com/davidiommi/3D-CycleGan-Pytorch-MedImaging. The training and inference workflows can be referenced from the link above.
### 2. Train Semi-supervised model



## Stage 2

## Stage 3

## Inference

## Evaluation

## Results

| Model name       |  DICE  |    NSD   |
| ---------------- | :----: | :------: |
|       ours       | 81.60% |  89.83   |


## Contributing


## Acknowledgement


