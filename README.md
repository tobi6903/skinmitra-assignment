
# Cat vs Dog Image Classifier

## Overview
This project is a deep learning-based image classification system which uses CNNs(Convolutional Neural Networks) that can identify whether an uploaded image contains a **cat**, a **dog**, or is **uncertain** . The system uses a ResNet-18 model pretrained on ImageNet, fine-tuned to classify between two classes: cats and dogs. 

A **Streamlit** web app interface is provided to allow users to upload an image and see the classification result along with the confidence score.

---

## How It Works
1. **Model**:
   - A ResNet-18 model is used .
   - The final fully connected layer is modified to output probabilities for two classes: **cat** and **dog**.
   - During training, the model was fine-tuned using transfer learning.
   - The model is saved in a file named `model.pth`.

2. **Web Application**:
   - Built with Streamlit for an interactive user interface.
   - Users can upload an image, and the app will display:
     - The uploaded image.
     - The predicted class label (cat, dog, or uncertain).
       
3. **Uncertain Class**:
   - If the model’s confidence score for an image is below a threshold , the image is classified as "uncertain."

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/tobi6903/skinmitra-assignment.git
cd https://github.com/tobi6903/skinmitra-assignment.git
```

### 2. Install Dependencies
Ensure you have Python 3.12+ installed. Install the required packages using:
```bash
pip install -r requirements.txt
```

**Dependencies:**
- `torch` and `torchvision`: For deep learning and model inference.
- `streamlit`: For building the web application.
- `Pillow`: For image processing.

### 3. Add the Model File
Place the pre-trained model file (`model.pth`) in the root directory of the project.

---

## Running the Application

### 1. Start the Streamlit App
Run the following command in the terminal:
```bash
streamlit run app.py
```

### 2. Upload an Image
- Open the URL provided by Streamlit in your browser (e.g., `http://localhost:8501`).
- Use the file uploader to upload an image (supported formats: `.jpg`, `.jpeg`, `.png`).
- The app will display:
  - The uploaded image.
  - The predicted class label (cat, dog, or uncertain).
  - The confidence score.

