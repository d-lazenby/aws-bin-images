import torch
import torch.nn as nn
import torchvision.models as models
import os
import torchvision.transforms as transforms
from PIL import Image
import io
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

request_content_type = "image/jpeg"

def net():
    
    model = models.resnet18(pretrained=True)

    # Freezes convolutional layers of model
    for param in model.parameters():
        param.requires_grad = False

    # Number of features in the output of the pretrained model
    num_features = model.fc.in_features

    # Output 5 channels
    model.fc = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 5)
    )

    return model

def model_fn(model_dir):
    print(f"In model_fn. Model directory is - {model_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = net().to(device)
    with open(os.path.join(model_dir, "model.pt"), "rb") as f:
        print("Loading model")
        model.load_state_dict(torch.load(f, map_location=device))
        logger.info('Model loaded successfully')
    model.eval()
    return model

def input_fn(request_body, request_content_type=request_content_type):
    logger.info('In input fn')
    # process an image uploaded to the endpoint
    logger.info('Deserializing the input data.')
    return Image.open(io.BytesIO(request_body))

def predict_fn(input_object, model):
    logger.info('In predict fn')
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    logger.info("Transforming input")
    input_object = test_transform(input_object)
    with torch.no_grad():
        logger.info("Calling model")
        prediction = model(input_object.unsqueeze(0))
        logger.info("Model called")
    return prediction