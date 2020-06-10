# A function to separate each TIF frame which consists of 30 frames total

# Import Packages
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load data
dir_data = './datasets'

name_label = "train-labels.tif"
name_input = "train-volume.tif"

img_label = Image.open(os.path.join(dir_data, name_label))
img_input = Image.open(os.path.join(dir_data, name_input))

ny, nx = img_label.size
nframe = img_label.n_frames

# Out of 30 frames, we store 24 frames (80%) into training 
# and 3 frames (10%) into validation, and remainig 3 frames (10%) into testing
nframe_train = 24
nframe_val = 3
nframe_test = 3

# Set the directory where the data will be stored
dir_save_train = os.path.join(dir_data,'train')
dir_save_val = os.path.join(dir_data,'val')
dir_save_test = os.path.join(dir_data,'test')

# Create directory
if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)
if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)
if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)
    
# Randomly store train/val/test data
id_frame = np.arange(nframe)
np.random.shuffle(id_frame)

# %%
# Save the dataset in each folder

# Train Folder
offset_n_frame = 0

for i in range(nframe_train):
    img_label.seek(id_frame[i + offset_n_frame])
    img_input.seek(id_frame[i + offset_n_frame])
    
    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)
    
    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), input_)
    
# Valid Folder
# make sure to change this to only apply remaining numbers to valid and test sets
offset_n_frame += nframe_train

for i in range(nframe_val):
    img_label.seek(id_frame[i + offset_n_frame])
    img_input.seek(id_frame[i + offset_n_frame])
    
    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)
    
    np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), input_) 
    
# Test Folder
# make sure to change this to only apply remaining numbers to valid and test sets
offset_n_frame += nframe_val

for i in range(nframe_test):
    img_label.seek(id_frame[i + offset_n_frame])
    img_input.seek(id_frame[i + offset_n_frame])
    
    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)
    
    np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), input_)         
    
# %%
# Load the sample image   
plt.subplot(121)
plt.imshow(label_, cmap='gray')
plt.title('label')

plt.subplot(122)
plt.imshow(input_, cmap='gray')
plt.title('input')