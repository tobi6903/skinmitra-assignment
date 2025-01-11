import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
import os
from torch import nn

# Model class
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)  

    def forward(self, x):
        return self.model(x)

# Data preprocessing 
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

# Load the model and weights
def load_model(model_path="model.pth"):
    model = ImageClassifier()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))  
    model.to(device) 
    model.eval()  
    return model


# Make predictions
def predict(model, image_path, device, threshold=0.92):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device) 
    
    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probs, 1)
        
        class_names = ["cat", "dog"]
        predicted_label = class_names[predicted_class.item()]
        confidence_score = confidence.item()
        
        # If confidence is low, classify as "uncertain"
        if confidence_score < threshold: 
            predicted_label = "uncertain"
        
        return predicted_label, confidence_score

# Streamlit Interface
def main():
    st.title("Cat vs Dog Classifier")
    st.write("Upload an image of a cat or dog, and I'll tell you what I think!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("model.pth")
    model = model.to(device)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
    
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Classifying..."):
            label, confidence = predict(model, uploaded_file, device)
            st.write(f"Prediction: {label}")
            st.write(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main()
