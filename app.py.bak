import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="AI-Snaily - Multi-YOLO Object Detection",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_model(model_path):
    """Load YOLO model with caching"""
    return YOLO(model_path)

def get_available_models():
    """Get available models from weights folder"""
    weights_dir = "weights"
    if not os.path.exists(weights_dir):
        return {}
    
    models = {}
    for file in os.listdir(weights_dir):
        if file.endswith('.pt'):
            model_name = file.replace('.pt', '')
            models[model_name] = os.path.join(weights_dir, file)
    
    return models

def main():
    st.title("🔍 AI-Snaily - Multi-YOLO Object Detection App")
    st.markdown("Support for YOLOv8, YOLOv10, and YOLOv11 models")
    
    # Sidebar for model configuration
    st.sidebar.header("Model Configuration")
    
    # Get available models from weights folder
    available_models = get_available_models()
    
    if not available_models:
        st.error("No models found in 'weights' folder. Please add .pt files to the weights directory.")
        st.stop()
    
    # Model selection from weights folder only
    selected_model_name = st.sidebar.selectbox(
        "Select Model",
        list(available_models.keys()),
        help="Choose from your custom models in the weights folder"
    )
    
    model_path = available_models[selected_model_name]
    st.sidebar.success(f"Selected: {selected_model_name}")
    st.sidebar.info(f"Model path: {model_path}")
    
    # Detection parameters
    confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01)
    iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.01)
    
    # Source selection
    st.sidebar.header("Input Source")
    source_option = st.sidebar.selectbox(
        "Choose Input Source",
        ["Upload Image", "Upload Video", "Webcam"]
    )
    
    try:
        # Load model
        with st.spinner("Loading model..."):
            model = load_model(model_path)
        
        if source_option == "Upload Image":
            handle_image_detection(model, confidence, iou_threshold)
        elif source_option == "Upload Video":
            handle_video_detection(model, confidence, iou_threshold)
        elif source_option == "Webcam":
            handle_webcam_detection(model, confidence, iou_threshold)
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")

def handle_image_detection(model, confidence, iou_threshold):
    """Handle image upload and detection"""
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp']
    )
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("Detection Results")
            
            # Run detection
            with st.spinner("Running detection..."):
                results = model(image, conf=confidence, iou=iou_threshold)
            
            # Display results
            annotated_image = results[0].plot()
            st.image(annotated_image, use_container_width=True)
            
            # Display detection statistics
            detections = results[0].boxes
            if detections is not None:
                st.write(f"**Objects detected:** {len(detections)}")
                
                # Class counts
                if hasattr(detections, 'cls'):
                    classes = detections.cls.cpu().numpy()
                    class_names = [model.names[int(cls)] for cls in classes]
                    class_counts = {}
                    for name in class_names:
                        class_counts[name] = class_counts.get(name, 0) + 1
                    
                    st.write("**Detection Summary:**")
                    for class_name, count in class_counts.items():
                        st.write(f"- {class_name}: {count}")

def handle_video_detection(model, confidence, iou_threshold):
    """Handle video upload and detection"""
    uploaded_video = st.file_uploader(
        "Upload a video",
        type=['mp4', 'avi', 'mov', 'mkv']
    )
    
    if uploaded_video is not None:
        # Save uploaded video
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.read())
        
        st.video("temp_video.mp4")
        
        if st.button("Run Detection on Video"):
            process_video(model, "temp_video.mp4", confidence, iou_threshold)

def handle_webcam_detection(model, confidence, iou_threshold):
    """Handle webcam detection"""
    st.write("### Webcam Detection")
    
    if st.button("Start Webcam Detection"):
        st.info("Webcam detection requires running locally. Use the code below:")
        
        webcam_code = '''
# Run this code locally for webcam detection
import cv2
from ultralytics import YOLO

model = YOLO("your_model.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame, conf=0.25)
    annotated_frame = results[0].plot()
    
    cv2.imshow("YOLO Detection", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
        '''
        
        st.code(webcam_code, language='python')

def process_video(model, video_path, confidence, iou_threshold):
    """Process video with YOLO detection"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))
    
    progress_bar = st.progress(0)
    frame_placeholder = st.empty()
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results = model(frame, conf=confidence, iou=iou_threshold)
        annotated_frame = results[0].plot()
        
        # Write frame
        out.write(annotated_frame)
        
        # Update progress
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        
        # Display current frame (every 30th frame to avoid lag)
        if frame_count % 30 == 0:
            frame_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
    
    cap.release()
    out.release()
    
    st.success("Video processing complete!")
    st.video("output_video.mp4")

if __name__ == "__main__":
    main()