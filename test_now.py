import torch
import os
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

# 1. Model Configuration
device = torch.device("cuda")
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('best_model.pth', weights_only=True))
model = model.to(device)
model.eval()

# 2. Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Folder Inference Function
def test_folder(folder_path):
    classes = ['Cat', 'Dog']
    print(f"🔍 Starting image inspection in folder: {folder_path}\n" + "-"*30)
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"❌ Folder '{folder_path}' does not exist!")
        return

    for img_name in os.listdir(folder_path):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, img_name)
            try:
                img = Image.open(img_path).convert('RGB') # Convert to handle transparent PNGs
                img_t = transform(img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(img_t)
                    _, predicted = torch.max(outputs, 1)
                
                print(f"📷 Image: {img_name}  ==> Prediction: {classes[predicted.item()]}")
            except Exception as e:
                print(f"⚠️ Issue with image {img_name}: {e}")

# Run testing on the 'data/test' folder
test_folder('data/test')