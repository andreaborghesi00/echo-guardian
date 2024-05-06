from torch.utils.data import Dataset
import SimpleITK as sitk
import radiomics
import torch
import numpy as np

class RadiomicsDataset(Dataset):
    def __init__(self, img_mask_paths, labels):
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

        self.extractor = radiomics.featureextractor.RadiomicsFeatureExtractor()
        self.extractor.disableAllFeatures()
        self.extractor.enableFeatureClassByName('firstorder')
        self.extractor.enableFeatureClassByName('shape2D')
        self.extractor.enableFeaturesByName(glcm=glcm_feats)
        self.extractor.enableFeatureClassByName('gldm')
        self.extractor.enableFeatureClassByName('glrlm')
        self.extractor.enableFeatureClassByName('glszm')
        self.extractor.enableFeatureClassByName('ngtdm')

        self.length = len(labels)
        self.labels = torch.tensor(labels.tolist())
        self.img_mask_paths = img_mask_paths

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        image = sitk.ReadImage(self.img_mask_paths[idx][0], sitk.sitkInt32)
        mask = sitk.ReadImage(self.img_mask_paths[idx][1], sitk.sitkInt32)
        features = self.extractor.execute(image, mask, voxelBased=False, label=255)
        features_values = torch.tensor([float(features[key]) for key in features if key.startswith('original_')]) # che coglioni, che famo preprocessiamo tutte le immagini?

        return features_values, self.labels[idx]