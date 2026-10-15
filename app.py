import streamlit as st
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import cv2
import numpy as np
from PIL import Image

st.title("Mask R-CNN Object Detection")
st.write("Upload an image below to run your trained Mask R-CNN model and view the bounding boxes.")

# Load Model cached so it doesn't reload on every click
@st.cache_resource
def load_model():
    num_classes = 2  # Adjust according to your dataset classes
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    # Load your model weights file
    checkpoint = torch.load("mask_rcnn_best.pth", map_location=torch.device('cpu'))
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    return model

with st.spinner("Loading Mask R-CNN model... Please wait."):
    model = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Run Detection"):
        with st.spinner("Detecting objects..."):
            image_np = np.array(image)
            image_tensor = torchvision.transforms.functional.to_tensor(image_np).unsqueeze(0)
            
            with torch.no_grad():
                predictions = model(image_tensor)
                
            pred = predictions[0]
            annotated_image = image_np.copy()
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            
            threshold = 0.5
            for i in range(len(pred['boxes'])):
                score = pred['scores'][i].item()
                if score > threshold:
                    box = pred['boxes'][i].cpu().numpy().astype(int)
                    cv2.rectangle(annotated_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    text = f"{score:.2f}"
                    cv2.putText(annotated_image, text, (box[0], box[1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            st.image(annotated_image, caption="Detected Output with Bounding Boxes", use_column_width=True)
