# Model Load / Save Functions

# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import numpy as np
import os
import random

## Model save function 
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        
    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               './%s/model_epoch%d.pth' % (ckpt_dir, epoch))
    
# Model Load function
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch
    
    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key = lambda f: int(''.join(filter(str.isdight, f))))
    
    dict_model = torch.load('./%s/%s' % (ckpt_dir, ckpt_lst[-1]))
    
    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])
    
    return net, optim, epoch   