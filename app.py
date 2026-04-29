import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 1. إعدادات الصفحة
st.set_page_config(page_title="Eslam Nawar  - Deep Layer Inspector", layout="wide")
st.title("🔬 Eslam Nawar : Neural Network Layer Inspector")
st.write("Understand how the AI evolves its vision from simple lines to complex features across ResNet18 layers.")

device = torch.device("cpu")

@st.cache_resource
def load_model():
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load('best_model.pth', map_location=device, weights_only=True))
    model.eval()
    return model

model = load_model()

# 2. وظيفة استخراج خرائط الملامح (Feature Maps)
def get_feature_maps(model, input_tensor, layer_name):
    feature_maps = []
    target_layer = dict([*model.named_modules()])[layer_name]
    
    def hook_fn(module, input, output):
        feature_maps.append(output)
        
    handle = target_layer.register_forward_hook(hook_fn)
    model(input_tensor)
    handle.remove()
    
    return feature_maps[0]

# 3. الواجهة والمعالجة
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("Upload Image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption="Input Image", use_container_width=True)
        layer_to_inspect = st.selectbox(
            "Select Layer to Inspect:",
            ["layer1", "layer2", "layer3", "layer4"],
            index=0
        )

    if st.button("Inspect Internal Thinking"):
        # الجزء الأول: التوقع (Prediciton)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            conf, pred = torch.max(probs, 0)
            class_names = ['Cat', 'Dog']
            
        st.subheader(f"Analysis Result: {class_names[pred.item()]} ({conf.item()*100:.2f}%)")
        
        # الجزء الثاني: استخراج وعرض الـ Feature Maps
        with st.spinner(f"Extracting features from {layer_to_inspect}..."):
            f_maps = get_feature_maps(model, input_tensor, layer_to_inspect)
            
            fig, axes = plt.subplots(4, 4, figsize=(10, 10))
            for i, ax in enumerate(axes.flat):
                if i < f_maps.shape[1]:
                    f_map = f_maps[0, i, :, :].detach().numpy()
                    ax.imshow(f_map, cmap='viridis')
                    ax.axis('off')
            
            st.pyplot(fig)
            
            # الجزء الثالث: التعليق المخصص لكل Layer
            st.markdown("### 💡 Layer Explanation:")
            
            if layer_to_inspect == "layer1":
                st.info("**Layer 1 (Edge Detection):** In this early stage, the AI acts like a sketch artist. It looks for basic edges, lines, and textures. You can still recognize the general shape of the pet because the filters are focusing on low-level visual details.")
            
            elif layer_to_inspect == "layer2":
                st.info("**Layer 2 (Pattern Recognition):** The AI begins to combine lines into basic patterns like curves and corners. The image starts to look more abstract as the network filters out background noise to focus on pet-specific textures.")
            
            elif layer_to_inspect == "layer3":
                st.info("**Layer 3 (Part Assembly):** Features become more complex. The AI is now looking for significant parts like ears, paws, or the curve of a tail. The resolution decreases as the 'semantic' meaning of the image increases.")
            
            elif layer_to_inspect == "layer4":
                st.info("**Layer 4 (High-Level Concepts):** This is the final stage before decision-making. The AI sees 'concepts' rather than pixels. Each bright spot represents a complex feature (like a specific eye shape) that confirms if the image is a Cat or a Dog.")