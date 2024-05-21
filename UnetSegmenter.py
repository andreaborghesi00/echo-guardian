import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import pickle
import cv2


class UnetSegmenter():
    def __init__(self, model_path='model.pth'):
        
        checkpoint = torch.load(model_path)
        self.model = checkpoint['model']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = pickle.load(open('./models/scaler_segmentation.pkl', 'rb'))
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def load_model(self, model_path):
        self.model = torch.load(model_path)
    
    def predict(self, image):
            """
            Predicts the class label for the given image.

            Parameters:
            image (numpy.ndarray or PIL.Image or str or torch.Tensor): The input image to be classified.

            Returns:
            torch.Tensor: The predicted class label.

            Raises:
            TypeError: If the image type is not supported.
            """
            # check type of image
            
            image = np.uint8(image)
            image = cv2.resize(image, (256, 256))
            image = self.scaler.transform(image)
            print(image.shape)
            image = torch.tensor(image).float().unsqueeze(0).unsqueeze(0).to(self.device)
            
            # resize image
            
            
            
            self.model.eval()
            
            with torch.no_grad():
                mask = self.model(image)
                mask = torch.sigmoid(mask)
                mask = torch.round(mask)
                mask = mask.int().squeeze().cpu().numpy()
                
            return mask
