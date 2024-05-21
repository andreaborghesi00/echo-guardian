import torch
import numpy as np
from PIL import Image

class NNClassifier():
    def __init__(self, model_path='model.pth'):
        self.load_model(model_path)

    def load_model(self, model_path):
        self.model = torch.load(model_path)
    
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
            prediction = None


            # check type of image
            if isinstance(image, np.ndarray):
                prediction = self.model(NNClassifier.extract_radiomics(image, mask))
            elif isinstance(image, Image):
                image = np.array(image)
                prediction = self.model(NNClassifier.extract_radiomics(image, mask))
            elif isinstance(image, str):
                image = Image.open(image).convert("L")
                image = np.array(image)
                prediction = self.model(NNClassifier.extract_radiomics(image, mask))
            elif isinstance(image, torch.Tensor):
                image = image.numpy()
                prediction = self.model(NNClassifier.extract_radiomics(image, mask))
            else:
                raise TypeError("Image type not supported")
            return prediction
    
    @staticmethod
    def extract_radiomics(image, mask):
        if image.shape != mask.shape:
            raise ValueError("Image and mask must have the same shape")
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be a numpy array")
        pass