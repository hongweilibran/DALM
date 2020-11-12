## Description of the datasest
The link to load the dataset: https://drive.google.com/drive/folders/1dgp5AVuzO9W-RlRtqOwHMXlX5bCX1zQ-?usp=sharing. 

The structure of fold is:  \ 
-DALM_datasets \
 --data  \ 
 --label \
where the folder named *data* contains 13 pre-processed samples (i.e. raw image array) in our experiment and the folder named *label* inlcudes the corresponding segmentation masks.  \ 

In each file in *data*, for example, the file named 'T7.mat', the dimension of the  is [96, 96, 254], where the last dimention is the number of channels. The first three dimentions represents the 3 channels of RGB images. Peusdo codes in python: 
```
from scipy.io import loadmat
import numpy as np
image = loadmat('DALM_datasets/data/T7.mat')
RGB_images = image[:, :, 0:3]  #get the RGB images
HS_images = image[:, :, 3:]   #get the hyperspectral images
```
