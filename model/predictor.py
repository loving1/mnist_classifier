import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io

class MNISTPredictor:
    def __init__(self, model_path="model/mnist_model_scripted.pt"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the TorchScript model
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    def preprocess_image(self, image_bytes):
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        
        # Apply transforms
        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(self.device)
    
    def predict(self, image_bytes):
        # Preprocess the image
        tensor = self.preprocess_image(image_bytes)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get the predicted class and its probability
            pred_prob, pred_class = torch.max(probabilities, 1)
            
        return {
            "prediction": pred_class.item(),
            "confidence": float(pred_prob.item())
        }
    
    def predict_from_array(self, array):
        # Convert numpy array to tensor and normalize
        tensor = torch.from_numpy(array).float().unsqueeze(0).unsqueeze(0)
        tensor = (tensor - 0.1307) / 0.3081
        tensor = tensor.to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get the predicted class and its probability
            pred_prob, pred_class = torch.max(probabilities, 1)
            
            # Get all class probabilities
            all_probs = probabilities[0].tolist()
            
        return {
            "prediction": pred_class.item(),
            "confidence": float(pred_prob.item()),
            "probabilities": all_probs
        }