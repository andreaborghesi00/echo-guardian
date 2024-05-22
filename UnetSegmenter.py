import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import pickle
import cv2


class UnetSegmenter():
    def __init__(self, model_path='DeepLabV3Plus_resnet34_lr_0.0001_epochs_100_actual_model.pth'):
        self.load_model(model_path)
        self.scaler = pickle.load(open('./models/scaler_segmentation.pkl', 'rb'))

    def load_model(self, model_path):
        
        checkpoint = torch.load(model_path)
        self.model = checkpoint['model']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        self.model.eval()
        self.model.to(self.device)
    
    def predict(self, image):
            """
            Predicts the class label for the given image.

            Parameters:
            image (numpy.ndarray or PIL.Image or str or torch.Tensor): The input image to be classified.
            The size of the image should be 256x256, no color channels.

            Returns:
            torch.Tensor: The predicted class label.

            Raises:
            TypeError: If the image type is not supported.
            """
            
            if isinstance(image, torch.Tensor):
                if image.device != 'cpu':
                    image = image.cpu().numpy()
                else:
                    image = image.numpy()
                
            if isinstance(image, Image.Image):
                image = np.array(image)
                image = np.uint8(image)
                        
            # 5 is a value that should not be reached by the image if it is already scaled, so we can use it as a flag to not scale the image
            if not image.max() <= 5:
                image = image / 255.0
            
            if image.shape[0] != 256 or image.shape[1] != 256:
                image = cv2.resize(image, (256, 256))
                
            
            image = torch.tensor(image).float().unsqueeze(0).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                mask = self.model(image)
                mask = torch.sigmoid(mask)
                mask = torch.round(mask)
                mask = mask.int().squeeze().cpu().numpy()
            return mask
