import h5py
from matplotlib import pyplot as plt
import matplotlib.image
from PIL import Image
from tqdm import tqdm
import numpy as np
import re
import random
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import string
import sys, os
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append( os.path.join(current_path, '../..') ) 
import utils
import matplotlib.colors as clrs
import colorsys
from classification.dataset import WbcDataset
# cmap = colormap.CellfaceStdCMap
def check_data_type(data):
    '''
    check data type, if data is numpy array, convert it to torch tensor
        Parameters:
            data: numpy array or torch tensor
        Returns:
            torch tensor        
    
    '''
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, torch.Tensor):
        return data 
    else:
        return torch.tensor(data)
        

class CellsDataset(Dataset):
    '''Cells dataset
        Parameters: 
            X: numpy array or torch tensor, images
            y: numpy array or torch tensor, labels
        Returns:
            x: torch tensor, image and labels
    '''
    def __init__(self, X, y):
      
        self.x_data = check_data_type(X)

        self.y_data = check_data_type(y)
        
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, index):
        x = self.x_data[index]
        y = self.y_data[index]
        return x, y


def data_augmentation(root_dir, png_save_root_dir, save_dataset=False, dataset_save_path='augmentation_data_555.pt',
                      CellsDataset=Dataset, num_augmentation=4, suffix='png'):
    """
    Augment a dataset of cell images with random rotations.

    Parameters:
    - root_dir (str): Directory path containing the original dataset.
    - png_save_root_dir (str): Directory where augmented images will be saved in PNG format.
    - save_dataset (bool): A flag to decide whether to save the augmented dataset or not. Defaults to False.
    - dataset_save_path (str): Path where the augmented dataset will be saved if save_dataset is True. Defaults to 'augmentation_data_555.pt'.
    - CellsDataset (class): The dataset class to be used. Defaults to a generic Dataset.
    - num_augmentation (int): Number of augmented images to produce for each original image. Defaults to 4.
    - suffix (str): File extension for reading images. Defaults to 'png'.

    Workflow:
    1. Defines a dictionary to convert cell names to integer labels and vice-versa.
    2. Reads the original dataset using the WbcDataset class.
    3. Converts the read images to a specific format suitable for transformations.
    4. Defines the augmentation transformation - a random rotation up to 180 degrees.
    5. Applies the transformation to each image in the dataset, num_augmentation times, accumulating the results.
    6. If save_dataset is True, the augmented dataset is saved to the specified path.
    7. Defines a colormap for the visualization of the cell images.
    8. Saves each augmented image in the png_save_root_dir directory with a unique randomized name, prefixed with the label of the cell.

    Returns:
    None. The function performs operations in-place and saves the results to the specified directories.

    Notes:
    - The function is designed specifically for cell images and uses predefined label names like 'wbc', 'rbc', etc.
    - The images are saved with random names to ensure uniqueness and are prefixed with their respective cell labels.
    - A specific colormap, CellfaceStdCMap, is used for visualization purposes while saving the augmented images.
    """
    
    cells2int_dict = {'wbc':0, 'rbc':1, 'plt':2, 'agg':3, 'oof':4}
    int2cells_dict = {v: k for k, v in cells2int_dict.items()}
    
    dataset_train_augment = WbcDataset(dir=root_dir, split='all',
                                    transform=None, download=False, need_label=True, resize=False, need_feature=False)
    images_list = []
    labels_list = []
    for i in range(len(dataset_train_augment)):
        image, label, _ = dataset_train_augment[i]
        images_list.append(image)
        labels_list.append(label)
        print(np.shape(image))
        print(np.shape(image))
        print(np.shape(image))
    images = np.stack(images_list, axis=0)
    labels = np.array(labels_list)
    
    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Grayscale(),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180, fill=255),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=(0,)*3, std=(255,)*3)
    ])

    
    labels = torch.tensor(labels)
    images = np.asarray(images)
    original_dataset = CellsDataset(images, labels)


    # Apply data augmentation transformations to generate augmented samples
    new_images = torch.empty(0)
    new_labels = torch.empty(0)

    num_augmentation = 4
    for _ in range(num_augmentation):
        augmented_data = torch.stack([transforms.functional.pil_to_tensor(data_transform(image)) for image in original_dataset.x_data], dim=0)
        new_images = torch.concat([new_images, augmented_data], dim=0)
        new_labels = torch.concat([new_labels, labels])
    new_images = torch.moveaxis(new_images, (0,1,2,3), (0,3,1,2)).to(torch.uint8)
    augmented_dataset = CellsDataset(new_images, new_labels)
    
    label_cnt = {}
    if save_dataset:
        torch.save(augmented_dataset, dataset_save_path)
    # define colormap
    CellfaceStdNorm = clrs.Normalize(vmin=-4, vmax=14, clip=True)
    CellfaceStdCMap = clrs.LinearSegmentedColormap.from_list(
        "CellFace Standard",
        [
            # Position               R     G     B     A
            (CellfaceStdNorm(-4.0), [0.65, 0.93, 1.00, 1.0]),  # Air bubbles
            (CellfaceStdNorm(0.0), [1.00, 0.97, 0.96, 1.0]),  # Background
        ] + [
            (
                CellfaceStdNorm(2 + p * (14 - 2)),
                [max(min(val, 1.0), 0.0) for val in list(
                    colorsys.hsv_to_rgb(
                        (280 - 90 * p) / 360,  # Hue: From Pink to Purple
                        0.5 + 1 * p,  # Saturation: Pastel to fully saturated
                        (1 - p) ** 2,  # Value: From Bright to Black
                    )
                )]
                + [1.0],
            )
            for p in np.linspace(0.0, 1.0, 20)
        ],
    )
    cmap = CellfaceStdCMap
    for image, label in augmented_dataset:
        
        ### 0001 0002-like rename ###
        # label = int2cells_dict[label.item()]
        # if label in label_cnt:
        #     label_cnt[label] += 1
        # else:
        #     label_cnt[label] = 1
        # cnt = label_cnt[label]
            
        image_np = image.detach().cpu().numpy()
        
        ### random name given ###
        # Generate a random name for the image
        random_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        label = int2cells_dict[label.item()]
        new_filename = f"{label}_{random_name}.png"
        # Get the directory path and create it if it doesn't exist
        destination_path = png_save_root_dir
        os.makedirs(destination_path, exist_ok=True)
        # Check if the new filename already exists in the destination folder
        while os.path.exists(os.path.join(destination_path, new_filename)):
            random_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
            new_filename = f"{label}_{random_name}.png"
        # Rename and move the image to the destination folder
        save_path = os.path.join(destination_path, new_filename)
        plt.imsave(save_path, cmap=cmap, arr=image_np)

if __name__ == '__main__':
    
    
    root_dir = r'W:\prediction'
    # save_dir_rel = "samples/sample01"
    png_save_root_dir = r'D:\aug_png'

    data_augmentation(root_dir, png_save_root_dir, save_datset=False, dataset_save_path='augmentation_data_555.pt', num_augmentation=4)