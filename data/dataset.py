import os
from PIL import Image
import gdown
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from .augments import augmentations_train, preprocess_train
from rich import print
import random
import yaml

# Decorator change directory to 'data' before function call
def in_pkg_dir(func):
    def wrapper(*args, **kwargs):
        os.chdir('./data')
        ret = func(*args, **kwargs)
        os.chdir('..')
        return ret
    return wrapper


# Download dataset from Google Drive 
# https://drive.google.com/file/d/1Wsw5VTVLMawzbcep47zkscXawd7gfHzT/view?usp=sharing
def download_dataset():
    gdown.download('https://drive.google.com/uc?id=1Wsw5VTVLMawzbcep47zkscXawd7gfHzT', 
                   './data.zip', fuzzy=True)
    os.system('unzip data.zip')
    os.system('rm data.zip')

# Load dataset from ./data/raw by saving the images and masks in ./data/train and ./data/val
@in_pkg_dir
def load_dataset():
    if not os.path.exists('data'):
        download_dataset()
    if not os.path.exists('train'):
        os.mkdir('train')
        os.mkdir('train/images')
        os.mkdir('train/masks')
        os.mkdir('val')
        os.mkdir('val/images')
        os.mkdir('val/masks')

        if not os.path.exists('split.yaml'):
            image_list = os.listdir(os.path.join('data', 'images'))
            random.shuffle(image_list)
            train_list = image_list[:int(len(image_list)*0.8)]
            val_list = image_list[int(len(image_list)*0.8):]
            with open('split.yaml', 'w') as f:
                yaml.dump({'train': train_list, 'val': val_list}, f)
        else:
            with open('split.yaml', 'r') as f:
                split = yaml.load(f, Loader=yaml.FullLoader)
                train_list = split['train']
                val_list = split['val']

        for image in train_list:
            os.system(f'cp data/images/{image} train/images')
            os.system(f'cp data/masks/{image} train/masks')
        for image in val_list:
            os.system(f'cp data/images/{image} val/images')
            os.system(f'cp data/masks/{image} val/masks')
        
# Define dataset class -- assume use of Albumentations for transforms
class DocSegmentDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(os.path.join('data', root_dir, 'images'))
        self.masks = os.listdir(os.path.join('data', root_dir, 'masks'))
        self.impath = os.path.join('data', root_dir, 'images')
        self.maskpath = os.path.join('data', root_dir, 'masks')
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = Image.open(os.path.join(self.impath, self.images[idx]))
        mask = Image.open(os.path.join(self.maskpath, self.masks[idx]))
        image, mask = np.array(image).astype(np.float32), np.array(mask)

        mask[mask!=0] = 1
        image /= 255

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        mask = mask.to(torch.long)
        return image, mask

# Return train and validation dataloaders
def build_dataloaders(batch_size=32, num_workers=2):
    load_dataset()

    train_dataset = DocSegmentDataset('train', transform=augmentations_train)  
    val_dataset = DocSegmentDataset('val', transform=preprocess_train)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers)

    print(f'[bold green]Train dataset loaded: {len(train_dataset)}[/bold green]')
    print(f'[bold green]Validation dataset loaded: {len(val_dataset)}[/bold green]')

    return train_loader, val_loader

