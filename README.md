# Joint Unsupervised Domain Adaptation and Semi-Supervised Learning for Multi-Sequence MR Abdominal Organ Segmentation
[![flow](https://github.com/Ho-Garfield/-FLARE2024_solution_he/blob/main/1.png)](flow)


## Environments and Requirements

- Ubuntu 22.04.2 LTS
- Intel(R) Xeon(R) Silver 4210R CPU @ 2.40GHz, 8 × 32GB; 2400MT/s, NVIDIA GeForce RTX 4090 24G
- cuda 11.8
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
Note that the suffix of the LLD dataset should meet one of the following formats:
```
FLARE24/
├── CT/
│   ├── images/
│   │   ├── ..._0000.nii.gz
│   │   └── ...
│   ├── labels/
│   │   ├── ....nii.gz
│   │   └── ...
├── MRI/
│   ├── LLD/
│   │   ├── ..._C-pre_0000.nii.gz
│   │   ├── ..._C--A_0000.nii.gz
│   │   ├── ..._C--V_0000.nii.gz
│   │   ├── ..._C--Delay_0000.nii.gz
│   │   ├── ..._InPhase_0000.nii.gz
│   │   ├── ..._OutPhase_0000.nii.gz
│   │   ├── ..._T2WI_0000.nii.gz
│   │   ├── ..._DWI_0000.nii.gz
│   │   └── ...
│   ├── AMOS/
│   │   ├── ..._0000.nii.gz
│   │   └── ...


```
## Preprocessing

Click preprocess to enter the preprocessing window.

[![flow](https://github.com/Ho-Garfield/-FLARE2024_solution_he/blob/main/3.jpg)](flow)


### 1. Resampling. 
Click the standard.py button, modify the parameters, and then click the Run button to repeatedly execute the program for CT's images, labels, as well as AMOS and LLD.

[![flow](https://github.com/Ho-Garfield/-FLARE2024_solution_he/blob/main/4.jpg)](flow)

### 2. Patient position readjustment.
Click the `flip_mri_with_inphase.py` button, modify the parameters, and then click the `Run` button.

[![flow](https://github.com/Ho-Garfield/-FLARE2024_solution_he/blob/main/5.jpg)](flow)

### 3. Adjust grayscale range.
Click the `gray2255.py` button, modify the parameters, and then click the `Run` button.

[![flow](https://github.com/Ho-Garfield/-FLARE2024_solution_he/blob/main/6.jpg)](flow)


### 4. Translation registration.(CT<->MRI T1W）
Click the `registration_ct_mri.py` button, modify the parameters, and then click the `Run` button.

[![flow](https://github.com/Ho-Garfield/-FLARE2024_solution_he/blob/main/7.jpg)](flow)


## Stage 1
Click Stage1 to enter the Stage1 window.

[![flow](https://github.com/Ho-Garfield/-FLARE2024_solution_he/blob/main/s1.jpg)](flow)

### 1. 3D CycleGAN
In this project, the implementation of the 3D CycleGAN is based on the repository available at https://github.com/davidiommi/3D-CycleGan-Pytorch-MedImaging. The training and inference workflows can be referenced from the link above.
### 2. Train Semi-supervised model



## Stage 2
Click Stage2 to enter the Stage2 window.

[![flow](https://github.com/Ho-Garfield/-FLARE2024_solution_he/blob/main/s2.jpg)](flow)

## Stage 3
Click Stage3 to enter the Stage3 window.

[![flow](https://github.com/Ho-Garfield/-FLARE2024_solution_he/blob/main/s3.jpg)](flow)

## Inference
Click Stage3 to enter the Stage3 window.

Click the `predict.py` button, modify the parameters, and then click the `Run` button.

[![flow](https://github.com/Ho-Garfield/-FLARE2024_solution_he/blob/main/inference.jpg)](flow)

## Evaluation
https://github.com/JunMa11/FLARE/tree/main/FLARE24
## Results

| Model name       |  DICE  |    NSD   |
| ---------------- | :----: | :------: |
|       ours       | 81.60% |  89.83%  |


## Contributing


## Acknowledgement
We thank the contributors of public datasets.




