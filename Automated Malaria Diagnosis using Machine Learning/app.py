import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np

# --- 1. Define the LeNet Model Architecture ---
# This is a critical step: The architecture MUST be defined before loading the model.
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)  # Binary classification: Infected vs Uninfected

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 13 * 13)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --- 2. Load the Trained Model (Trusted Source) ---
@st.cache_resource
def load_model():
    try:
        model = torch.load(
            'trained_model.sav',
            map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
            weights_only=False,  # Ensure only weights are loaded, not the entire model object
        )
        model.eval()
        return model
    except FileNotFoundError:
        st.error("Error: 'trained_model.sav' not found. Please ensure it's in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Make sure the model file is a valid PyTorch model and the class definition matches.")
        st.stop()

model = load_model()

# --- 3. Define Image Preprocessing ---
preprocess = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- 4. Streamlit App Interface ---
st.title("🦠 Malaria Cell Classifier")
st.write("Upload a microscopic image of a blood cell to classify it as **Infected** or **Uninfected**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("Classify"):
        with st.spinner('Classifying...'):
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0).to('cpu')

            with torch.no_grad():
                output = model(input_batch)

            _, predicted_class = torch.max(output, 1)

            if predicted_class.item() == 1:
                st.error("Prediction: **Infected**")
                st.write("The model has detected signs of a malaria parasite in the cell.")
            else:
                st.success("Prediction: **Uninfected**")
                st.write("The model has classified the cell as uninfected.")