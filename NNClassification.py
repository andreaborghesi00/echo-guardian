import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import radiomics
import pickle
import torchvision.transforms as transforms
import SimpleITK as sitk
import cv2

class NNClassifier():
    def __init__(self, model_path='model.pth'):
        self.load_model(model_path)
        self.scaler = pickle.load(open('./models/scaler_classification.pkl', 'rb'))

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.model = checkpoint['model']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
    
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, image, mask):
            """
            Predicts the class label for the given image.

            Parameters:
            image (numpy.ndarray or PIL.Image or str or torch.Tensor): The input image to be classified.

            Returns:
            torch.Tensor: The predicted class label.

            Raises:
            TypeError: If the image type is not supported.
            """
            self.model.eval()
            if isinstance(image, Image.Image):
                image = np.array(image)
                mask = np.array(mask)
                
            if isinstance(image, torch.Tensor):
                image = image.numpy()
                mask = mask.numpy()

            if np.max(image) <= 1:
                image = image * 255
            if np.max(mask) <= 1:
                mask = mask * 255
            
            image = cv2.resize(np.array(image), (256, 256))
            mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

            
            features = self.extract_radiomics(image, mask)
            features = self.scaler.transform(np.array(features).reshape(1, -1))
            with torch.no_grad():
                prediction = self.model(torch.Tensor(features).to(self.device))

            return prediction
    
    @staticmethod
    def extract_radiomics(image, mask):
        if image.shape != mask.shape:
            raise ValueError(f"Image and mask must have the same shape: image shape is {image.shape}, mask shape is {mask.shape}")
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be a numpy array")
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
        extractor = radiomics.featureextractor.RadiomicsFeatureExtractor()
        extractor.disableAllFeatures()
        extractor.enableFeatureClassByName('firstorder')
        extractor.enableFeatureClassByName('shape2D')
        extractor.enableFeaturesByName(glcm=glcm_feats)
        extractor.enableFeatureClassByName('gldm')
        extractor.enableFeatureClassByName('glrlm')
        extractor.enableFeatureClassByName('glszm')
        extractor.enableFeatureClassByName('ngtdm')
        sitk_image = sitk.GetImageFromArray(image.astype(float))
        sitk_mask = sitk.GetImageFromArray(mask.astype(float))
        
        
        features = extractor.execute(sitk_image, sitk_mask, voxelBased=False, label=255)
        features_values = [float(features[key]) for key in features if key.startswith('original_')]
        return features_values