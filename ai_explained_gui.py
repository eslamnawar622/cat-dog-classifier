import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import customtkinter as ctk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 1. Model Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('best_model.pth', weights_only=True))
model = model.to(device).eval()

class ProAIExplainer(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("ResNet18 Journey Visualizer | Eslam|Nawar")
        self.geometry("1200x750")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # UI Layout: 2 Columns
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar (Control Panel) ---
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        self.logo_label = ctk.CTkLabel(self.sidebar, text="AXIS AI VISION", font=ctk.CTkFont(size=22, weight="bold"))
        self.logo_label.pack(pady=30, padx=20)

        self.btn_upload = ctk.CTkButton(self.sidebar, text="Upload Image", height=40, font=("Segoe UI", 14, "bold"), command=self.analyze)
        self.btn_upload.pack(pady=10, padx=20)

        self.divider = ctk.CTkLabel(self.sidebar, text="─" * 20, text_color="gray")
        self.divider.pack(pady=10)

        self.status_title = ctk.CTkLabel(self.sidebar, text="PROCESS INSIGHTS", font=("Segoe UI", 16, "bold"), text_color="#1f538d")
        self.status_title.pack(pady=10)

        self.desc_box = ctk.CTkTextbox(self.sidebar, width=240, height=350, font=("Segoe UI", 13), corner_radius=10)
        self.desc_box.pack(pady=10, padx=20)
        self.desc_box.insert("0.0", "System Ready...\nPlease upload a JPEG/PNG image to begin the multi-layer feature extraction process.")

        # --- Main Content (Visualization Area) ---
        self.main_view = ctk.CTkFrame(self, corner_radius=15, fg_color="#1a1a1a")
        self.main_view.grid(row=0, column=1, padx=25, pady=25, sticky="nsew")

    def analyze(self):
        file_path = filedialog.askopenfilename()
        if not file_path: return

        # Image Preprocessing
        img = Image.open(file_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_t = transform(img).unsqueeze(0).to(device)

        # Journey Extraction
        with torch.no_grad():
            l1 = model.relu(model.bn1(model.conv1(img_t)))
            l2 = model.layer2(model.layer1(model.maxpool(l1)))
            l3 = model.layer4(model.layer3(l2))
            
            outputs = model(img_t)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, pred = torch.max(probabilities, 1)

        self.render_visuals(img, l1, l2, l3, pred.item(), confidence.item() * 100)

    def render_visuals(self, img, l1, l2, l3, pred_idx, conf_score):
        for widget in self.main_view.winfo_children(): widget.destroy()

        fig, axes = plt.subplots(1, 4, figsize=(16, 6), facecolor='#1a1a1a')
        titles = ["Original Input", "Layer 1: Edges", "Layer 2: Textures", "Final Features"]
        layers = [img, l1, l2, l3]

        for i, ax in enumerate(axes):
            ax.axis('off')
            ax.set_title(titles[i], color="#ffffff", pad=15, fontsize=11, fontweight='bold')
            if i == 0:
                ax.imshow(img.resize((224, 224)))
            else:
                feat = torch.mean(layers[i][0], dim=0).cpu().numpy()
                ax.imshow(feat, cmap='inferno') # 'inferno' looks more professional/techy

        canvas = FigureCanvasTkAgg(fig, master=self.main_view)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        # Professional English Description
        label = "DOG" if pred_idx == 1 else "CAT"
        summary = f"INFERENCE REPORT\n" + "="*20 + "\n"
        summary += f"Result: {label}\n"
        summary += f"Confidence: {conf_score:.2f}%\n\n"
        summary += "TECHNICAL STAGES:\n"
        summary += "1. Input: 224x224 RGB matrix.\n\n"
        summary += "2. Edges: Low-level filters isolated high-frequency components (outlines).\n\n"
        summary += "3. Textures: Mid-level blocks synthesized edges into structural patterns.\n\n"
        summary += "4. Concepts: Deep layers abstracted the image into spatial feature maps for classification."

        self.desc_box.delete("0.0", "end")
        self.desc_box.insert("0.0", summary)

if __name__ == "__main__":
    app = ProAIExplainer()
    app.mainloop()