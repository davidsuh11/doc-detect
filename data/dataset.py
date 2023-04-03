import os
import pandas as pd
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

import torch
import albumentations as A
from torch.utils.data import Dataset, DataLoader
import numpy as np
from .augments import augmentations_train, preprocess_train
from rich import print
from rich.progress import Progress
RAW_DIR_NAME = 'raw'

# Decorator change directory to 'data' before function call
def in_top_dir(func):
    def wrapper(*args, **kwargs):
        os.chdir('./data')
        ret = func(*args, **kwargs)
        os.chdir('..')
        return ret
    return wrapper

# Delete all files in directory
def clear_dir(dir):
    for file in os.listdir(dir):
        os.remove(os.path.join(dir, file))

# Download raw dataset file from 
# #https://github.com/jchazalon/smartdoc15-ch1-dataset/releases/download/v2.0.0/frames.tar.gz
def download_dataset():
    if not os.path.exists(target:= os.path.join('./data', RAW_DIR_NAME)):
        os.mkdir(target)
    else:
        return False

    os.system(f'./data/download_data.sh')
    return True

def _load_annotation(file, df, split_ratio, prog, task):
    # Load annotation file ./data/raw/metadata.csv as csv in pandas
    filepath = os.path.join(RAW_DIR_NAME, file)

    current_frame = 1
    total_frames = len(os.listdir(filepath))
    

    for impath in os.listdir(filepath):
        target_dir = 'train' if current_frame < total_frames * split_ratio else 'val'

        # Load image
        im = Image.open(os.path.join(filepath, impath))
        local_impath = os.path.join(file, impath)

        # Load annotation
        annotation = df[df['image_path'] == local_impath]
        coords = [
            annotation['tl_x'], annotation['tl_y'],
            annotation['tr_x'], annotation['tr_y'],
            annotation['br_x'], annotation['br_y'],
            annotation['bl_x'], annotation['bl_y']
        ]
        coords = [int(x) for x in coords]

        # Create mask
        mask = Image.new('L', im.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(coords, fill=1, outline=1)

        # Save mask
        namedata = file.split('/')
        to_save = f'{namedata[0]}_{namedata[1]}_{impath}'

        mask.save(os.path.join(target_dir, 'masks', to_save))
        im.save(os.path.join(target_dir, 'images', to_save))
        current_frame += 1
    
    prog.update(task, advance=1)
        
# Load dataset from ./data/raw by saving the images and masks in ./data/train and ./data/val
@in_top_dir
def load_dataset(split_ratio=0.8, clear_existing=False):
    for dir in ['train', 'val']:
        if not os.path.exists(dir):
            os.mkdir(dir)
        if not os.path.exists(os.path.join(dir, 'images')):
            os.mkdir(os.path.join(dir, 'images'))
        if not os.path.exists(os.path.join(dir, 'masks')):
            os.mkdir(os.path.join(dir, 'masks'))

    if clear_existing:
        for dir in ['train', 'val']:
            clear_dir(os.path.join(dir, 'images'))
            clear_dir(os.path.join(dir, 'masks'))

    df = pd.read_csv(os.path.join(RAW_DIR_NAME, 'metadata.csv'))
    
    
    with Progress() as prog:
        task = prog.add_task("[red]Loading dataset...", total=5*5*5)
        for n in range(1,6):
            for j in range(1,6):
                for type in ['datasheet', 'letter', 'magazine', 'paper', 'tax']:
                    _load_annotation(f'background0{n}/{type}00{j}', df, split_ratio, prog, task)
    prog.stop_task(task)
    
    print(f'Dataset loaded\n Train: {len(os.listdir("train/images"))}\nVal: {len(os.listdir("val/images"))}')

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
def build_dataloaders(batch_size=4, num_workers=2, reload_dataset=False):
    if reload_dataset: load_dataset()
    train_dataset = DocSegmentDataset('train', transform=augmentations_train)  
    val_dataset = DocSegmentDataset('val', transform=preprocess_train)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers)

    return train_loader, val_loader

