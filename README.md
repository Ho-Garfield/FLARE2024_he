# Joint Unsupervised Domain Adaptation and Semi-Supervised Learning for Multi-Sequence MR Abdominal Organ Segmentation
[![flow](https://github.com/Ho-Garfield/-FLARE2024_solution_he/blob/main/1.png)](flow)


## Environments and Requirements

- Ubuntu 22.04.2 LTS
- Intel(R) Xeon(R) Silver 4210R CPU @ 2.40GHz, 8 × 32GB; 2400MT/s, NVIDIA GeForce RTX 4090 24G
- 11.8
- Python 3.9.0

To install requirements:

```setup
cd miccai_model
conda create -n miccai python=3.9
conda activate miccai
pip install -e .
```



## Dataset

https://www.codabench.org/competitions/2296/

## Preprocessing
### 1. Resampling. 

Modify the parameters in ```miccai_model/step1_standardized.py```:
```
input_dir = "data/origin_labels"
output_dir = "data/labels"
in_sub = ".nii.gz"
out_sub = ".nii.gz"
is_label = True #When true, use nearest-neighbor interpolation; when false, use cubic spline interpolation.
```
Run step1_standardized.py
```
python step1_standardized.py
```

### 2. Patient position readjustment.
   
Modify the parameters in ```miccai_model/step1_filip_mri_with_inphase.py```:
```
inphase_dir ="/DATA_16T/MICCAI/FLARE/FLARE24-Task3-MR/Training/LLD-MMRI-3984"
handle_dir = "/DATA_16T/MICCAI/FLARE/FLARE24-Task3-MR/Training/LLD-MMRI-3984"
out_dir = "/DATA_16T/MICCAI/Flip_resample"
```
Run step1_filip_mri_with_inphase.py
```
python step1_filip_mri_with_inphase.py
```

### 3. Adjust grayscale range.

Modify the parameters and code in ```miccai_model/step1_Normalize.py```:
```
input_dir = ""
out_dir =""
...
ptqdm(function = CTNormalization, iterable = img_paths, processes = 4, min = -160, max=240,out_dir=out_dir)
#ptqdm(function = MRINormalization, iterable = img_paths, processes = 4, out_dir=out_dir)
```
Run step1_Normalize.py
```
python step1_Normalize.py
```
### 4. Translation registration.(CT<->MRI T1W）
Create the CT_select and T1W directories, and place the data in the following format.
```
CT_select/
├── images/
│   ├── ct1_0000.nii.gz
│   ├── ct2_0000.nii.gz
│   ├── ...
|   └── ct498_0000.nii.gz
└── labels/
    ├── ct1.nii.gz
    ├── ct2.nii.gz
    ├── ...
    └── ct498.nii.gz
T1W/
├── 1_pre_0000.nii.gz
├── 2_pre_0000.nii.gz
├── 3_pre_0000.nii.gz
├── ...
└── 498_pre_0000.nii.gz

```
Modify the parameters in ```miccai_model/step1_registration_ct_mri.py```:
```
img_sub = "_0000.nii.gz"
lab_sub = ".nii.gz"

a0_path ="./CT_select/images/FLARE22_Tr_0001_0000.nii.gz"
a0_label_path ="./CT_select/labels/FLARE22_Tr_0001.nii.gz"
b0_path = "./T1W/MR745_6_C-pre_0000.nii.gz"

out_dir_mid = ""
out_dir_final = "miccai_model/3D-CycleGan-Pytorch-MedImaging-main/Data_folder/train"

input_images_dir="./CT_select/images"
input_labels_dir ="./CT_select/labels"
input_targets_dir="./T1W"

```
Run step1_registration_ct_mri.py
```
python step1_registration_ct_mri.py
```
This will generate a directory in the miccai_model/3D-CycleGan-Pytorch-MedImaging-main folder with the following structure.
```
├── Data_folder                   
|   ├── train              
|   |   ├── images             
|   |   |   ├── 0.nii              
|   |   |   ├── 1.nii
|   |   |   └── ...                    
|   |   └── labels            
|   |   |   ├── 0.nii             
|   |   |   ├── 1.nii
|   |   |   └── ...
|   |   └── image_labels            
|   |   |   ├── 0.nii             
|   |   |   ├── 1.nii
|   |   |   └── ...

```
## Stage 1
### 1. 3D CycleGAN
In this project, the implementation of the 3D CycleGAN is based on the repository available at https://github.com/davidiommi/3D-CycleGan-Pytorch-MedImaging. The training and inference workflows can be referenced from the link above.
### 2. Train Semi-supervised model
Create the ```miccai_model/data/``` folder. 

Use the trained 3D CycleGAN to generate T1W data corresponding to the CT data in ```miccai_model/3D-CycleGan-Pytorch-MedImaging-main/Data_folder/train/images```. 

Place these files along with the files from the ```T1W/``` directory into ```miccai_model/data/images```(. 

At the same time, place the files from ```miccai_model/3D-CycleGan-Pytorch-MedImaging-main/Data_folder/train/images_label``` into ```miccai_model/data/labels```. 

The structure of the data directory is as follows:
```
├── data                   
|   ├── images                          
|   |   ├── 0.nii              
|   |   ├── 1.nii
|   |   ├── ...
|   |   ├── 1_pre_0000.nii.gz
|   |   ├── 2_pre_0000.nii.gz
|   |   ├── 3_pre_0000.nii.gz
|   |   ├── ...
|   |   └── 498_pre_0000.nii.gz                 
|   └── labels            
|   |   ├── 0.nii             
|   |   ├── 1.nii
|   |   └── ...
Modify the parameters in ```miccai_model/semi/config.py```:

```
Run train.py
```
python train.py
```



## Stage 2

## Stage 3

## Inference

1. edit miccai_model/semi/predict.py
```
model_path = ""
image_folder=""
predict_folder= ""
```
2.```python predict.py```
## Evaluation

## Results

| Model name       |  DICE  |    NSD   |
| ---------------- | :----: | :------: |
|       ours       | 81.60% |  89.83   |


## Contributing


## Acknowledgement


