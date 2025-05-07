import io
import os
import torch
import torch.nn as nn
import timm

from torchvision import models, transforms
from flask import Flask, request, jsonify
from PIL import Image, UnidentifiedImageError
from PIL.ImageFile import ImageFile


class Predictor:
    def __init__(self, model_name: str = 'efficientnet-b0', weights_path: str = None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        match model_name:
            case "mobilenetv3-large":
                model_ft = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT, progress=True)
                model_ft.classifier[-1] = nn.Linear(1280, 2)
            case "mobilenetv3-small":
                model_ft = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT, progress=True)
                model_ft.classifier[-1] = nn.Linear(1024, 2)
            case "efficientnet-b0":
                model_ft = models.efficientnet_b0(weights=None if weights_path else 'IMAGENET1K_V1', progress=True)
                model_ft.classifier[-1] = nn.Linear(1280, 2)
            case "resnet18":
                model_ft = models.resnet18(weights='IMAGENET1K_V1', progress=True)
                model_ft.fc = nn.Linear(512, 2)
            case "efficientnet-b3":
                model_ft = models.efficientnet_b3(weights='IMAGENET1K_V1', progress=True)
                model_ft.classifier[-1] = nn.Linear(1536, 2)
            case "squeezenet":
                model_ft = models.squeezenet1_0(weights='IMAGENET1K_V1', progress=True)
                model_ft.classifier[1] = nn.Conv2d(512, 2, kernel_size=(1,1), stride=(1,1))
                model_ft.num_classes = 2
            case "densenet121":
                model_ft = models.densenet121(weights='IMAGENET1K_V1', progress=True)
                model_ft.classifier = nn.Linear(1024, 2)
            case "vit-tiny":
                model_ft = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=2)
            case "convnext-tiny":
                model_ft = timm.create_model('convnext_tiny', pretrained=True, num_classes=2)
            case _:
                raise ValueError(f"Unknown model: {model_name}")
        self.model = model_ft
        self._image_transforms = transforms.Compose([
            transforms.Resize(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        if weights_path:
            self._load_weights(weights_path)
        
    
    def _load_weights(self, weights_path: str):
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found at {weights_path}")
        
        checkpoint = torch.load(weights_path, weights_only=False, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()        
    
    def _transform_image(self, image: ImageFile):
        image = image.convert('RGB')
        image = self._image_transforms(image)
        image = image.unsqueeze(0)
        image = image.to(self.device)
        
        return image
    
    def predict(self, image: ImageFile):
        with torch.no_grad():
            image = self._transform_image(image)
            outputs = self.model(image)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            prob_no_glaucoma = probabilities[0][0].item() * 100
            prob_glaucoma = probabilities[0][1].item() * 100
            predicted_class = predicted.item()
            
            return predicted_class, prob_no_glaucoma, prob_glaucoma


weights_path = os.getenv('WEIGHTS_PATH', r'train\checkpoints\efficientnet-b0_checkpoint.pth')

app = Flask(__name__)
predictor = Predictor(model_name='efficientnet-b0', weights_path=weights_path)


@app.route('/predict-glaucoma', methods=['POST'])
def predict_glaucoma():
    if 'image' not in request.files:
        return jsonify({"error": "no image provided"}), 400
    
    image_file = request.files['image']
    image_bytes = image_file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except UnidentifiedImageError:
        return jsonify({'error': 'Incorrect image'}), 400
    
    try:
        predicted_class, prob_no_glaucoma, prob_glaucoma = predictor.predict(image)
    except Exception as e:
        print(e)
        return jsonify({'error': 'Internal server error'}), 500
    
    response = jsonify({
        "is_glaucoma": predicted_class == 1,
        "prob_glaucoma": prob_glaucoma,
        "prob_no_glaucoma": prob_no_glaucoma,
    })
    
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    
    return response, 200


if __name__ == '__main__':
    app.run('0.0.0.0', 8081)
    