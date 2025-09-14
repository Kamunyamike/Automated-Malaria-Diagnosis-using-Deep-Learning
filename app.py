import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Set page configuration
st.set_page_config(
    page_title="Malaria Detection System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
IMG_HEIGHT = 128
IMG_WIDTH = 128

# Define the LeNet model architecture (matching the trained model)
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.fc1 = nn.Linear(32 * (IMG_HEIGHT // 4) * (IMG_WIDTH // 4), 128)
        
        self.fc2 = nn.Linear(128, 1)  # Binary classification, 1 logit
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = x.view(-1, 32 * (IMG_HEIGHT // 4) * (IMG_WIDTH // 4))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Define transforms for preprocessing
def get_transforms():
    return A.Compose([
        A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

@st.cache_resource
def load_model():
    """Load the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNet()
    
    try:
        model.load_state_dict(torch.load('malaria_classification_model.pth', map_location=device))
        model.to(device)
        model.eval()
        return model, device
    except FileNotFoundError:
        st.error("❌ Model file 'malaria_classification_model.pth' not found! Please ensure the model is trained and saved.")
        return None, device

def preprocess_image(image):
    """Preprocess the uploaded image"""
    # Convert PIL image to numpy array
    image_np = np.array(image)
    
    # Convert RGB to BGR (OpenCV format)
    if len(image_np.shape) == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Apply transforms
    transform = get_transforms()
    transformed = transform(image=image_np)
    image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
    
    return image_tensor

def predict_malaria(model, image_tensor, device):
    """Make prediction on the image"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        probability = torch.sigmoid(output).item()
        
        # Convert to percentage
        parasitized_prob = probability * 100
        uninfected_prob = (1 - probability) * 100
        
        prediction = "Parasitized" if probability > 0.5 else "Uninfected"
        confidence = max(parasitized_prob, uninfected_prob)
        
        return prediction, confidence, parasitized_prob, uninfected_prob

def main():
    # Header
    st.title("🔬 Malaria Detection System")
    st.markdown("---")
    
    # Sidebar with information
    with st.sidebar:
        st.header("📋 About this System")
        st.markdown("""
        This system uses a trained LeNet Convolutional Neural Network to detect malaria parasites in blood slide images.
        
        **How it works:**
        1. Upload a blood slide image
        2. The AI model analyzes the image
        3. Get prediction: Parasitized or Uninfected
        4. View confidence scores
        
        **Supported formats:** PNG, JPG, JPEG
        """)
        
        st.markdown("---")
        st.header("🔧 Model Information")
        
        # Display device information
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            st.success(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            st.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            st.info("💻 Running on CPU")
        
        st.info(f"Image Size: {IMG_WIDTH} × {IMG_HEIGHT}")
        st.info("Model: LeNet CNN")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Blood Slide Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a blood slide image for malaria detection"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=250)
            
            # Add some spacing
            st.markdown("---")
            
            # Prediction button
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    # Load model
                    model, device = load_model()
                    
                    if model is not None:
                        # Preprocess image
                        image_tensor = preprocess_image(image)
                        
                        # Make prediction
                        prediction, confidence, parasitized_prob, uninfected_prob = predict_malaria(
                            model, image_tensor, device
                        )
                        
                        # Store results in session state for display in col2
                        st.session_state.prediction = prediction
                        st.session_state.confidence = confidence
                        st.session_state.parasitized_prob = parasitized_prob
                        st.session_state.uninfected_prob = uninfected_prob
                        st.session_state.analysis_complete = True
    
    with col2:
        st.header("📊 Analysis Results")
        
        if hasattr(st.session_state, 'analysis_complete') and st.session_state.analysis_complete:
            # Display results
            prediction = st.session_state.prediction
            confidence = st.session_state.confidence
            parasitized_prob = st.session_state.parasitized_prob
            uninfected_prob = st.session_state.uninfected_prob
            
            # Main prediction
            if prediction == "Parasitized":
                st.error(f"🦠 **PARASITIZED** - Malaria detected!")
            else:
                st.success(f"✅ **UNINFECTED** - No malaria detected")
            
            st.metric("Confidence Level", f"{confidence:.1f}%")
            
            # Detailed probabilities
            st.markdown("### 📈 Detailed Probabilities")
            
            # Create probability bars
            st.markdown("**Parasitized (Malaria Present):**")
            st.progress(parasitized_prob / 100)
            st.text(f"{parasitized_prob:.1f}%")
            
            st.markdown("**Uninfected (No Malaria):**")
            st.progress(uninfected_prob / 100)
            st.text(f"{uninfected_prob:.1f}%")
            
            # Warning based on confidence
            if confidence < 70:
                st.warning("⚠️ Low confidence prediction. Consider consulting a medical professional.")
            elif confidence > 90:
                st.info("ℹ️ High confidence prediction.")
            
            # Medical disclaimer
            st.markdown("---")
            st.markdown("""
            **⚠️ Medical Disclaimer:**
            This is an AI-assisted diagnostic tool and should not replace professional medical diagnosis. 
            Always consult healthcare professionals for medical decisions.
            """)
        else:
            st.info("👆 Upload an image and click 'Analyze Image' to see results here.")
            
            # Display sample images info
            st.markdown("### 🖼️ Sample Images")
            st.markdown("""
            This system works best with:
            - Clear blood cell images
            - Good lighting and focus
            - Images similar to the training data
            - PNG, JPG, or JPEG format
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🔬 Malaria Detection System | Powered by Deep Learning & PyTorch</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

