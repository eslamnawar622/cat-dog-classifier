import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 1. إعداد الجهاز والموديل
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('best_model.pth', weights_only=True))
model = model.to(device).eval()

# 2. تحويل الصورة وتجهيزها
def get_journey(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    # --- رحلة البيانات داخل الطبقات ---
    
    # الطبقة الأولى (المعالجة الأولية للحواف)
    x = model.conv1(img_tensor)
    x = model.bn1(x)
    layer1_out = model.relu(x)

    # الطبقة العميقة (استخراج الميزات المعقدة)
    x = model.maxpool(layer1_out)
    layer2_out = model.layer1(x)
    layer3_out = model.layer2(layer2_out)
    
    return img, layer1_out, layer3_out

# 3. دالة الرسم
def plot_journey(img, l1, l3):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # العرض الأول: المدخلات (Input)
    axes[0].imshow(img.resize((224, 224)))
    axes[0].set_title("1. Input Image (Pixels)\n224x224 RGB", fontsize=12, color='blue')
    axes[0].axis('off')

    # العرض الثاني: Hidden Layer 1 (Edges)
    # بناخد متوسط الـ 64 فلتر عشان نشوف الموديل ركز على إيه في البداية
    feat1 = torch.mean(l1[0], dim=0).detach().cpu().numpy()
    axes[1].imshow(feat1, cmap='magma')
    axes[1].set_title("2. Early Hidden Layer\n(Detecting Edges & Shapes)", fontsize=12, color='green')
    axes[1].axis('off')

    # العرض الثالث: Deeper Layer (Abstract Features)
    # هنا الموديل بيبدأ يشوف ملامح "حيوان" مش مجرد خطوط
    feat3 = torch.mean(l3[0], dim=0).detach().cpu().numpy()
    axes[2].imshow(feat3, cmap='viridis')
    axes[2].set_title("3. Deep Hidden Layer\n(Abstract Concept Features)", fontsize=12, color='red')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

# شغل الرحلة (حط مسار صورة من اللي عندك في الـ test)
image_to_check = 'data/test/12461.jpg' 
try:
    original, low_feat, high_feat = get_journey(image_to_check)
    plot_journey(original, low_feat, high_feat)
except Exception as e:
    print(f"❌ حصل مشكلة: {e}")