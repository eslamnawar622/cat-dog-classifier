import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import numpy as np
import time

# 1. Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"🚀 Execution Engine: {torch.cuda.get_device_name(0)}")
else:
    print("🚀 Execution Engine: CPU")

# 2. Data Preparation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Ensure your data path is correct (./data)
full_dataset = datasets.ImageFolder(root='./data', transform=transform)

# Selecting a random subset (100 images)
indices = np.random.choice(len(full_dataset), 100, replace=False)
small_dataset = Subset(full_dataset, indices)
train_loader = DataLoader(small_dataset, batch_size=10, shuffle=True)

# 3. Model Architecture (ResNet18)
model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, 2) 
model = model.to(device)

# 4. Hyperparameters
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 5. Training Loop
epochs = 5 
print(f"\n--- Starting Training on {len(small_dataset)} images ---")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Metrics Calculation
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    duration = time.time() - start_time

    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}% | Duration: {duration:.2f}s")

# 6. Model Saving
torch.save(model.state_dict(), 'best_model.pth')
print("\n✨ Success: Model saved as 'best_model.pth'")