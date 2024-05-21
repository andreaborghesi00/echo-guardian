from torch.utils.data import Dataset
import cv2
import torch
import json
from tqdm import tqdm
import re
from PIL import Image
import numpy as np
import pandas as pd

class SegmentationDataset(Dataset):
    def __init__(self, img_mask_paths, labels, transform = None, json_exclude_path=None, exclusion_class="cnn"):
        self.transform = transform
        self.img_mask_paths = img_mask_paths

        json_data = {}
        #load json
        if json_exclude_path is not None:
            with open(json_exclude_path, 'r') as f:
                json_data = json.load(f)

        self.exclusion_list = []
        for i in tqdm(range(len(labels))):
            mask_path = self.img_mask_paths[i][1]
            
            if json_exclude_path is not None and 'benign' in mask_path and int(re.findall(r'\d+', mask_path)[0]) in json_data[exclusion_class]['benign']:
                self.exclusion_list.append(i)
                continue
            elif json_exclude_path is not None and 'malignant' in mask_path and int(re.findall(r'\d+', mask_path)[0]) in json_data[exclusion_class]['malignant']:
                self.exclusion_list.append(i)
                continue

        self.img_mask_paths = [path for i, path in enumerate(img_mask_paths) if i not in self.exclusion_list]

    def __len__(self):
        return len(self.img_mask_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_mask_paths[idx][0]
        mask_path = self.img_mask_paths[idx][1]
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask / 255.0
        
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask'].long().unsqueeze(0)
        
        
        

        return img, mask

