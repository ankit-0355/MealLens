import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from ultralytics import YOLO

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Paths and model types
sam_checkpoint = "D:\\projectworkspace\\fooddetection\\sam_vit_h_4b8939.pth"  # Make sure this checkpoint file is in your project directory
model_type = "vit_h"

# Load SAM model
@st.cache_resource
def load_sam_model():
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator

# Load YOLOv8 model
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")  # Replace with your trained food model if available

# Load models
mask_generator = load_sam_model()
yolo_model = load_yolo_model()

# Streamlit UI
st.title("🍱 Food Segmentation & Detection Dashboard")
st.markdown("Upload a food image with multiple items (e.g., apple & banana), and see segmentation + labeling results.")

uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Convert to numpy
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)

    # Segment with SAM
    st.info("🔍 Running SAM for segmentation...")
    masks = mask_generator.generate(img_np)
    st.success(f"Found {len(masks)} segments.")

    # Folder setup
    os.makedirs("output_segments", exist_ok=True)
    labels_detected = []

    # Loop over each segment
    for i, mask in enumerate(masks):
        seg = mask["segmentation"]
        x, y, w, h = cv2.boundingRect(seg.astype(np.uint8))
        cropped = img_np[y:y+h, x:x+w]

        # Save temp file for YOLO input
        temp_path = f"segment_{i}.jpg"
        cv2.imwrite(temp_path, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))

        # Run YOLOv8 on cropped segment
        results = yolo_model(temp_path)

        for box in results[0].boxes:
            label = results[0].names[int(box.cls)]
            labels_detected.append(label)

            # Save segment in folder named by label
            label_folder = os.path.join("output_segments", label)
            os.makedirs(label_folder, exist_ok=True)
            out_path = os.path.join(label_folder, f"{uploaded_file.name}_{i}.jpg")
            cv2.imwrite(out_path, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))

    # Display labels
    if labels_detected:
        st.markdown("### ✅ Detected Ingredients:")
        for label in set(labels_detected):
            st.write(f"- {label}: {labels_detected.count(label)} time(s)")
        st.success("Segments saved to 'output_segments' folder.")
    else:
        st.warning("No labels detected. Try another image or ensure your YOLO model is trained for food.")
