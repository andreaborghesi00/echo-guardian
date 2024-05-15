from torch.utils.data import Dataset
import cv2
from torch import tensor
import json
from tqdm import tqdm
import re

class SegmentationDataset(Dataset):
    def __init__(self, img_mask_paths, labels, augmentation=None, transform = None, json_exclude_path=None, exclusion_class="cnn"):
        self.augmentation = augmentation
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
        
        if self.augmentation and self.transform:
            # img = (self.transform(self.augmentation(img)))
            # mask = (self.transform(self.augmentation(mask)))
            augmented = self.transform(self.augmentation(image=img, mask=mask))
            print('Augmentation and transform of idx: {} done', idx)
        
        return augmented, mask

