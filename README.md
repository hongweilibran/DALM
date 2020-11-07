# DALM: Deep learning-enabled Accurate Layer Mapping of 2D materials
This is a public repository for deep learning-based accurate segmentation of 2D materials. The codes were mainly developed by Mr. Zhutong Jiang (Email: zhutongj@gmail.com) in the starting stage, and were further polished by Mr. Hongwei Li (Email: hongwei.li@tum.de) and Mr. Xingchen Dong (xingchen.dong@tum.de). 
![Framework](./framework.png)

<b>Input</b>: Hyperspectral images (2+1 D) and RGB images (2D) \
<b>Output</b>: Multi-class segmentation map of 2D materials \
<b>Model</b>: A two-stream convolutional neural network that fuses the dual-modality information

Specifically, in our work the dimensions of the inputs are <b>[96, 96, 128, 1]</b>, and <b>[96, 96, 3]</b> for hyperspectral and RGB images respectively.  
The key architecture is defined in 
```
model.py
```
