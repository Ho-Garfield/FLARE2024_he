# Joint Unsupervised Domain Adaptation and Semi-Supervised Learning for Multi-Sequence MR Abdominal Organ Segmentation
[![flow](https://github.com/Ho-Garfield/-FLARE2024_solution_he/blob/main/1.png)](flow)

## CT to T1W image translation 
In this project, the implementation of the 3D CycleGAN based on the repository available at https://github.com/davidiommi/3D-CycleGan-Pytorch-MedImaging. 

## Environments and Requirements

- Ubuntu 22.04.2 LTS
- Intel(R) Xeon(R) Silver 4210R CPU @ 2.40GHz, 8 × 32GB; 2400MT/s, NVIDIA GeForce RTX 4090 24G
- 11.8
- Python 3.9.0

To install requirements:

```setup
cd miccai_model
pip install -e .
```



## Dataset

https://www.codabench.org/competitions/2296/

## Preprocessing


## Training


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


