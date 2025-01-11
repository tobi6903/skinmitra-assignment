# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# Custom Dataset Class for loading images
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Define the model(Pretrained)
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2) 

    def forward(self, x):
        return self.model(x)

# Prepare DataLoader
def prepare_dataloader(data_dir, transform):
    categories = ["cats", "dogs"] 
    image_paths = []
    labels = []

    for label, category in enumerate(categories):
        category_dir = os.path.join(data_dir, category)
        for image_name in os.listdir(category_dir):
            if image_name.endswith(('.jpg', '.png', '.jpeg')):
                image_paths.append(os.path.join(category_dir, image_name))
                labels.append(label)
    
 
    train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42)
    
    train_dataset = CustomDataset(train_paths, train_labels, transform)
    val_dataset = CustomDataset(val_paths, val_labels, transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader

# Training Loop
def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs) # Forward Pass
            loss = criterion(outputs, labels)
            
            # Backward pass 
            loss.backward()
            optimizer.step() # Optimizer step
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_preds / total_preds * 100
        print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_accuracy:.2f}%")
        
        validate_model(model, val_loader, device)

    # Save the model 
    torch.save(model.state_dict(), "model.pth")

# Validation loop
def validate_model(model, val_loader, device):
    model.eval()
    correct_preds = 0
    total_preds = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)
    
    val_accuracy = correct_preds / total_preds * 100
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

# Prediction function 
def predict(model, image_path, device, threshold=0.95):
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
        
        # If confidence is low, classify them  as "uncertain"
        if confidence_score < threshold: 
            predicted_label = "uncertain"
        
        return predicted_label, confidence_score

# Data preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])


train_loader, val_loader = prepare_dataloader("/kaggle/input/cat-and-dog/training_set/training_set", transform)

model = ImageClassifier()
train_model(model, train_loader, val_loader, num_epochs=10)

# Make prediction
image_path = "/kaggle/input/cat-and-dog/test_set/test_set/cats/cat.4001.jpg"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label, confidence = predict(model, image_path, device)
print(f"Predicted label: {label}, Confidence: {confidence:.2f}")
