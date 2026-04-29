import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog

# 1. إعداد الموديل (نفس الإعدادات اللي مرنتها)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('best_model.pth', weights_only=True))
model = model.to(device)
model.eval()

# 2. تحويلات الصور
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. دالة التوقع
def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        # حساب الاحتمالات عشان نشوف الموديل واثق قد إيه (Softmax)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        prob, predicted = torch.max(probabilities, 1)
    
    classes = ['Cat', 'Dog']
    return classes[predicted.item()], prob.item() * 100

# 4. بناء الواجهة
root = tk.Tk()
root.title("Eslam Nawar's AI Classifier") # مستوحى من اسمك المسجل
root.geometry("600x500")

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # عرض الصورة
        img = Image.open(file_path)
        img.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        panel.config(image=img_tk)
        panel.image = img_tk
        
        # التوقع
        label, confidence = predict_image(file_path)
        
        # تحديث النتيجة
        result_label.config(text=f"Result: {label}", fg="#00FF00" if label == "Dog" else "#FF00FF")
        conf_label.config(text=f"Confidence: {confidence:.2f}%")

# تصميم العناصر
btn = tk.Button(root, text="Select Image", command=upload_image, font=("Arial", 12), bg="#2196F3", fg="white")
btn.pack(pady=20)

panel = tk.Label(root)
panel.pack()

result_label = tk.Label(root, text="Result: None", font=("Arial", 20, "bold"))
result_label.pack(pady=10)

conf_label = tk.Label(root, text="Confidence: 0%", font=("Arial", 12))
conf_label.pack()

root.mainloop()