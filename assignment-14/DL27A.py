# DL27A.py CS5173/6073 cheng 2023
# use pre-trained resnet50 for image classification
# based on PyTorch torchvision.models tutorial
# Usage: python DL27A.py

from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights

img = read_image("Chihuahua_dog_1.jpg")

# Step 1: Initialize model with the best available weights
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Step 4: Use the model and print the predicted category
prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")
