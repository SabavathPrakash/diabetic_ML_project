from flask import Flask, request, render_template
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn as nn
import os

app = Flask(__name__)

# Load your trained model
model = models.resnet34(pretrained=False)
model.fc = nn.Sequential(
    nn.Linear(512, 256),
    nn.Linear(256, 128),
    nn.Linear(128, 64),
    nn.Linear(64, 5)
)
model.load_state_dict(torch.load('subscribe.h5', map_location=torch.device('cpu')))
model.eval()

# Define transforms
transform = transforms.Compose([
    transforms.Resize([512, 512]),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Class labels
class_labels = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    if file:
        img = Image.open(file.stream).convert('RGB')
        img = transform(img).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = model(img)
            prediction = torch.argmax(output, dim=1).item()
            label = class_labels[prediction]

        return render_template('result.html', label=label)

if __name__ == '__main__':
    app.run(debug=True)
