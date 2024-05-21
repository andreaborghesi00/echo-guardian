import torch
import numpy as np
from PIL import Image

class UnetSegmenter():
    def __init__(self, model_path='model.pth'):
        self.load_model(model_path)

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
            self.model.eval()
            prediction = None

            # check type of image
            if isinstance(image, np.ndarray):
                prediction = self.model(image)
            elif isinstance(image, Image):
                prediction = self.model(np.array(image))
            elif isinstance(image, str):
                image = Image.open(image).convert("L")
                prediction = self.model(np.array(image))
            elif isinstance(image, torch.Tensor):
                prediction = self.model(image)
            else:
                raise TypeError("Image type not supported")
            return prediction
