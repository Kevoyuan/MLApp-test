import socket
from collections import namedtuple
import numpy as np
from PIL import Image
import os
from pathlib import Path
import torch
import re
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import json 

# Specify the directory path

def save_json(data, path):
    
    with open(path, "w") as outfile:
        json.dump(data, outfile)
    
def load_json(path):
    
    with open(path, 'r') as f:
        data = json.load(f)
        return data


def make_dir(path, parents=True, exist_ok=True):
    Path(path).mkdir(parents=parents, exist_ok=exist_ok)




def list_dir(root_dir, suffix=''):
# Walk through the directory tree
    file_paths = []
    for root, dirs, files in os.walk(root_dir):
        # Print the current directory
        # print('Directory:', root)

        # Print the files in the current directory
        if suffix == '':
            for file in files:
                file_paths.append(os.path.join(root, file))
                # print('File:', os.path.join(root, file))
        else:
            for file in files:
                if file.endswith(suffix):
                    file_paths.append(os.path.join(root, file))
    return file_paths
            

## dir root format accoring to the systems and users
def dataspace():
    '''
     Returns dir root format accoring to the systems and users.
            
            Returns:
                data_dirs (namedtuple): dir root format of data share and group share
    '''
    data_dir = None
    data_dirs = namedtuple('dataspace', ['data_dir', 'group_dir'])
    host_name = socket.gethostname()
    if host_name == r"DESKTOP-PE62B05":
        data_dir = r"W:"
        group_dir = r"Z:"
    
    if data_dir == None:
        assert("invalid hostname")
    
    return data_dirs(data_dir, group_dir)

def read_dataset_from_png(root_dir, suffix='.png', label_to_int=False, cells2int_dict=None, onlyX=False):
    
    file_paths = list_dir(root_dir, suffix)
    labels = []
    images = []
    for i in range(len(file_paths)):
        im = Image.open(file_paths[i]).convert('RGB')
        im = transforms.Grayscale()(im)
        # if np.asarray(im).shape == (96, 96):
        #     images.append(np.asarray(im))
        images.append(np.asarray(im))
        if not onlyX:
            file_name = os.path.basename(file_paths[i])[:-len(suffix)-1]
            lb = file_name[:3]  # Get the first three characters as label
            labels.append(lb)
    
    images = np.asarray(images)
    if not onlyX:
        labels = np.asarray(labels)
    
        if label_to_int:
            labels = [cells2int_dict[c] for c in labels]
    if not onlyX:
        return images, labels
    else :
        return images

if __name__ == '__main__':
    # datadirs = dataspace()
    # print(datadirs.data_dir, datadirs.group_dir)
    root_dir = r'W:\samples\prediction' 
    file_paths = list_dir(root_dir)
    # print(file_paths)