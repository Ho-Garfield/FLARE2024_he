# :triangular_flag_on_post: FLARE24 Task3 Solution
Our solution, 'he,' ranks first in both DSC and NSD in the final tests.

[![flow](https://github.com/Ho-Garfield/-FLARE2024_solution_he/blob/main/rank.png)](flow)
# :mag_right: Overview
Joint Unsupervised Domain Adaptation and Semi-Supervised Learning for Multi-Sequence MR Abdominal Organ Segmentation
[![flow](https://github.com/Ho-Garfield/-FLARE2024_solution_he/blob/main/1.png)](flow)


## :computer: Environments and Requirements

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



# :hospital: Dataset

https://www.codabench.org/competitions/2296/

## :door: Interactive window
```setup
python window.py
```
You can see that the window contains preprocessing, stage1, stage2, and stage3, as shown in the figure below.

[![flow](https://github.com/Ho-Garfield/-FLARE2024_solution_he/blob/main/2.png)](flow)

## :file_folder: Prepare the data
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
## :scissors: Preprocessing

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


## :one: Stage 1
Click Stage1 to enter the Stage1 window.

[![flow](https://github.com/Ho-Garfield/-FLARE2024_solution_he/blob/main/s1.jpg)](flow)

### 1. 3D CycleGAN
In this project, the implementation of the 3D CycleGAN is based on the repository available at https://github.com/davidiommi/3D-CycleGan-Pytorch-MedImaging. The training and inference workflows can be referenced from the link above.
### 2. Click the `gan_train.py` to train 3D CycleGAN

### 3. Rename GAN model and click the `gan_pred.py` to create Fake T1W sequence.

[![flow](https://github.com/Ho-Garfield/-FLARE2024_solution_he/blob/main/rename.jpg)](flow)

### 3. Click 'train.py' to train Semi-supervised model
### 3. Click 'predict.py' for predicting T1W sequences in LLD.


## :two: Stage 2
Click Stage2 to enter the Stage2 window.

[![flow](https://github.com/Ho-Garfield/-FLARE2024_solution_he/blob/main/s2.jpg)](flow)

## :three: Stage 3
Click Stage3 to enter the Stage3 window.

[![flow](https://github.com/Ho-Garfield/-FLARE2024_solution_he/blob/main/s3.jpg)](flow)

## :wrench: Inference
Click Stage3 to enter the Stage3 window.

Click the `predict.py` button, modify the parameters, and then click the `Run` button.

[![flow](https://github.com/Ho-Garfield/-FLARE2024_solution_he/blob/main/inference.jpg)](flow)

[Model download link](https://drive.google.com/file/d/10oppkUMdTFLdRi4KnoBvjpzLILPDG9N2/view?usp=drive_link)
## Evaluation
https://github.com/JunMa11/FLARE/tree/main/FLARE24
## :pushpin: Results

| Dataset Name	   |  DICE  |    NSD   |
| ---------------- | :----: | :------: |
|     Validation   | 81.60% |  89.83%  |
|       Test       |   ?    |    ?     |

Our method exhibits both excellent and stable performance in DSC and NSD, although it is slightly less efficient in terms of time and resource consumption.

[![flow](https://github.com/Ho-Garfield/-FLARE2024_solution_he/blob/main/result.png)](flow)
[![flow](https://github.com/Ho-Garfield/-FLARE2024_solution_he/blob/main/cost.png)](flow)

## Contributing


## Acknowledgement
We thank the contributors of public datasets.




