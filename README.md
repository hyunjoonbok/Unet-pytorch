# Unet-pytorch
U-Net in Pytorch
#### Title
U-Net: Convolutional Networks for Biomedical Image Segmentation

#### Discription 
From [Paper](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).
The u-net is convolutional network architecture for fast and precise segmentation of images. Up to now it has outperformed the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks. It has won the Grand Challenge for Computer-Automated Detection of Caries in Bitewing Radiography at ISBI 2015, and it has won the Cell Tracking Challenge at ISBI 2015 on the two most challenging transmitted light microscopy categories (Phase contrast and DIC microscopy) by a large margin (See also our annoucement).

#### Model Architecture
![image](https://lh3.googleusercontent.com/proxy/QPt20YDpodsDn101p8mjMDMKoRL4o3ss6lyROCIMoZllQWvgE-RFEgd_m2SSDEKA5dSJLzgwZ5FMbwhQxMcfo2VR3JI9_2CX9g)

## Data
- Dataset could be downloaded either from [this website](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) or [here](http://brainiac2.mit.edu/isbi_challenge/)(you would need to create log-in for this one)
- Please run **dataset.py** python file to create dataset that will be used in training. 

## Training
    $ python main.py --mode train \
                     --scope [scope name] \
                     --name_data [data name] \
                     --dir_data [data directory] \
                     --dir_log [log directory] \
                     --dir_checkpoint [checkpoint directory]
                     --gpu_ids [gpu id; '-1': no gpu, '0, 1, ..., N-1': gpus]
---
    $ python main.py --mode train \
                     --scope unet \
                     --name_data em \
                     --dir_data ./datasets \
                     --dir_log ./log \
                     --dir_checkpoint ./checkpoint
                     --gpu_ids 0







#### Paper: https://arxiv.org/abs/1505.04597
#### Test
