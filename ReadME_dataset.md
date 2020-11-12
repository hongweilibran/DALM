## Description of the datasest
The link to load the dataset: https://drive.google.com/drive/folders/1dgp5AVuzO9W-RlRtqOwHMXlX5bCX1zQ-?usp=sharing. 

The structure of the folders is:  \
-DALM_datasets \
&nbsp --data  \
&nbsp --label \
&nbsp --label_visualization \
where the folder named *data* contains 13 pre-processed samples (i.e. raw image array) in our experiment and the folder named *label* inlcudes the corresponding segmentation masks.  For visualzation, we also provide color images in the foler named *label_visualization* \

In each file in *data*, for example, the file named 'T7.mat', the dimension of the  is [96, 96, 254], where the last dimention of the dimentionis the number of channels. The first three dimentions represents the three channels of RGB images. Here are codes in python for demonstration to read the data, taking 'T7.mat' as an example: 
```
from scipy.io import loadmat
import numpy as np
image = loadmat('DALM_datasets/data/T7.mat')   #load .mat files in python
RGB_images = image[:, :, 0:3]  #get the RGB images
HS_images = image[:, :, 3:]   #get the hyperspectral images
```
The mannual labels are in PNG format and can be read using the codes as follows, taking 'L7.png' as an example: 
```
label = np.asarray(Image.open('DALM_datasets/data/L7.png'))  # get the labels with label = 0, 1, 2, 3, 4
```
where label = 0, 1, 2, 3, 4 represents background, mono-, bi-, tri-, and multi-layer thickness respectively. 
