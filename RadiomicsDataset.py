from torch.utils.data import Dataset
import SimpleITK as sitk
import radiomics
import albumentations as A
import torch
import numpy as np
from tqdm import tqdm
from sklearn.utils.validation import check_is_fitted
import json
import re

class RadiomicsDataset(Dataset):
    def __init__(self, img_mask_paths, labels, scaler=None, json_exclude_path=None, exclusion_class="classifier", transform = None):
        radiomics.logger.setLevel(40)
        glcm_feats = [ # i know it's annoying, and it took way too long to find, but this is how you exclude a feature from the extraction
            'Autocorrelation',
            'ClusterProminence',
            'ClusterShade',
            'ClusterTendency',
            'Contrast',
            'Correlation',
            'DifferenceAverage',
            'DifferenceEntropy',
            'DifferenceVariance',
            'Id',
            'Idm',
            'Idmn',
            'Idn',
            'Imc1',
            'Imc2',
            'InverseVariance',
            'JointAverage',
            'JointEnergy',
            'JointEntropy',
            'MCC',
            'MaximumProbability',
            # 'SumAverage',
            'SumEntropy',
            'SumSquares'
        ]

        self.transform = transform
        self.extractor = radiomics.featureextractor.RadiomicsFeatureExtractor()
        self.extractor.disableAllFeatures()
        self.extractor.enableFeatureClassByName('firstorder')
        self.extractor.enableFeatureClassByName('shape2D')
        self.extractor.enableFeaturesByName(glcm=glcm_feats)
        self.extractor.enableFeatureClassByName('gldm')
        self.extractor.enableFeatureClassByName('glrlm')
        self.extractor.enableFeatureClassByName('glszm')
        self.extractor.enableFeatureClassByName('ngtdm')

        self.scaler = scaler
        
        self.imarray = []
        self.maskarray = []
        
        self.img_mask_paths = img_mask_paths
        
        json_data = {}
        #load json
        if json_exclude_path is not None:
            with open(json_exclude_path, 'r') as f:
                json_data = json.load(f)                

        # populate them later in the loop
        self.rad_features = []
        self.labels = []

        for i in tqdm(range(len(labels))):
            image_path = self.img_mask_paths[i][0]
            mask_path = self.img_mask_paths[i][1]
            
            if json_exclude_path is not None and 'benign' in mask_path and int(re.findall(r'\d+', mask_path)[0]) in json_data[exclusion_class]['benign']:
                continue
            elif json_exclude_path is not None and 'malignant' in mask_path and int(re.findall(r'\d+', mask_path)[0]) in json_data[exclusion_class]['malignant']:
                continue

            image = sitk.ReadImage(image_path, sitk.sitkUInt8)
            mask = sitk.ReadImage(mask_path, sitk.sitkUInt8)
            
            self.imarray.append(sitk.GetArrayFromImage(image))
            self.maskarray.append(sitk.GetArrayFromImage(mask))
            
            features = self.extractor.execute(image, mask, voxelBased=False, label=255)
            features_values = [float(features[key]) for key in features if key.startswith('original_')]
            self.rad_features.append(features_values)
            self.labels.append(labels[i])


        if self.scaler is not None:
            try: 
                self.rad_features = self.scaler.transform(np.array(self.rad_features))        
            except:
                self.scaler = self.scaler.fit(np.array(self.rad_features))
                self.rad_features = self.scaler.transform(np.array(self.rad_features))        

        self.rad_features = torch.tensor(self.rad_features).float() 
        self.labels = torch.tensor(self.labels).unsqueeze(1).float() # Unsqueeze for BCE loss 2D input requirements

        self.length = len(self.rad_features)    
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.transform is None:
            feat = self.rad_features[idx]
        else:
            augmented = self.transform(image=self.imarray[idx], mask=self.maskarray[idx])
            try:
                features = self.extractor.execute(sitk.GetImageFromArray(augmented['image']), sitk.GetImageFromArray(augmented['mask']), voxelBased=False, label=255)
            except:
                print('Error in feature extraction:', self.img_mask_paths[idx][0], self.img_mask_paths[idx][1])
                return self.rad_features[idx], self.labels[idx]
            features_values = [float(features[key]) for key in features if key.startswith('original_')]

            if self.scaler is not None:
                feat = torch.tensor(self.scaler.transform(np.array(features_values).reshape(1, -1))).float()[0]
            else:
                feat = torch.tensor(features_values).float()

        #print (feat.shape, self.labels[idx].shape)
        #print(feat)
        return feat, self.labels[idx]