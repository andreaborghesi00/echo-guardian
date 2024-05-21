from torch.utils.data import Dataset
import cv2
import torch
import json
from tqdm import tqdm
import re
from PIL import Image
import numpy as np
import pickle

class SegmentationDataset(Dataset):
    def __init__(self, img_mask_paths, labels, transform = None, json_exclude_path=None, exclusion_class="cnn", scaler = None):
        self.transform = transform
        self.img_mask_paths = img_mask_paths
        self.scaler = scaler
        
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
        
        if self.scaler is not None:
            try:
                self.scaler.transform(np.zeros((256, 256)))
                print("Scaler already fitted")
            except Exception as e:
                for image_path, _ in self.img_mask_paths:
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (256, 256))
                    
                    self.scaler.partial_fit(img)
                pickle.dump(self.scaler, open('./models/scaler_segmentation.pkl', 'wb'))
            
        
    def __len__(self):
        return len(self.img_mask_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_mask_paths[idx][0]
        mask_path = self.img_mask_paths[idx][1]
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.uint8)
        mask = mask.astype(np.uint8)
        
        # check if mask max value is 255, if not, normalize it
        if mask.max() > 2:
            mask = mask / 255.0
        
        if self.scaler is not None:
            img = cv2.resize(img, (256, 256))
            mask = cv2.resize(mask, (256, 256))
            img = self.scaler.transform(img)
            
        
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask'].long().unsqueeze(0)
        
        return img.float(), mask.float()

