import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
import os
import time

# Page configuration
st.set_page_config(
    page_title="AI-Snaily - YOLOv5 Object Detection",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_yolov5_model(model_path):
    """Load YOLOv5 model with caching"""
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
    return model

def get_confidence_style(conf):
    """Get visual style based on confidence level"""
    if conf < 0.35:
        return {
            'color': (0, 0, 255),      # Red in BGR
            'thickness': 2,
            'line_type': 'dashed',
            'label': f'Low {conf:.2f}'
        }
    elif conf < 0.60:
        return {
            'color': (0, 0, 255),      # Red in BGR
            'thickness': 2,
            'line_type': 'dashed',
            'label': f'Low {conf:.2f}'
        }
    elif conf < 0.85:
        return {
            'color': (0, 255, 255),    # Yellow in BGR
            'thickness': 2,
            'line_type': 'dotted',
            'label': f'Mid {conf:.2f}'
        }
    else:
        return {
            'color': (0, 255, 0),      # Green in BGR
            'thickness': 2,
            'line_type': 'solid',
            'label': f'High {conf:.2f}'
        }

def generate_color_map(class_names):
    """Generate unique colors for each class"""
    colors = {}
    color_palette = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan
        (128, 0, 0),      # Maroon
        (0, 128, 0),      # Dark Green
        (0, 0, 128),      # Dark Blue
        (128, 128, 0),    # Olive
        (128, 0, 128),    # Purple
        (0, 128, 128),    # Teal
        (255, 128, 0),    # Orange
        (255, 0, 128),    # Pink
        (128, 255, 0),    # Lime
    ]
    for i, class_name in enumerate(class_names):
        colors[class_name] = color_palette[i % len(color_palette)]
    return colors

def draw_dashed_rectangle(image, pt1, pt2, color, thickness=2):
    """Draw dashed rectangle on image"""
    x1, y1 = pt1
    x2, y2 = pt2
    dash_length = 10
    gap_length = 5
    
    # Top line
    for x in range(x1, x2, dash_length + gap_length):
        end_x = min(x + dash_length, x2)
        cv2.line(image, (x, y1), (end_x, y1), color, thickness)
    
    # Bottom line
    for x in range(x1, x2, dash_length + gap_length):
        end_x = min(x + dash_length, x2)
        cv2.line(image, (x, y2), (end_x, y2), color, thickness)
    
    # Left line
    for y in range(y1, y2, dash_length + gap_length):
        end_y = min(y + dash_length, y2)
        cv2.line(image, (x1, y), (x1, end_y), color, thickness)
    
    # Right line
    for y in range(y1, y2, dash_length + gap_length):
        end_y = min(y + dash_length, y2)
        cv2.line(image, (x2, y), (x2, end_y), color, thickness)

def draw_dotted_rectangle(image, pt1, pt2, color, thickness=2):
    """Draw dotted rectangle on image"""
    x1, y1 = pt1
    x2, y2 = pt2
    dot_spacing = 5
    
    # Top line
    for x in range(x1, x2, dot_spacing):
        cv2.circle(image, (x, y1), thickness, color, -1)
    
    # Bottom line
    for x in range(x1, x2, dot_spacing):
        cv2.circle(image, (x, y2), thickness, color, -1)
    
    # Left line
    for y in range(y1, y2, dot_spacing):
        cv2.circle(image, (x1, y), thickness, color, -1)
    
    # Right line
    for y in range(y1, y2, dot_spacing):
        cv2.circle(image, (x2, y), thickness, color, -1)

def draw_rectangle(image, pt1, pt2, color, thickness=2, line_type='solid'):
    """Draw rectangle with specified line type"""
    if line_type == 'dashed':
        draw_dashed_rectangle(image, pt1, pt2, color, thickness)
    elif line_type == 'dotted':
        draw_dotted_rectangle(image, pt1, pt2, color, thickness)
    else:  # solid
        cv2.rectangle(image, pt1, pt2, color, thickness)

def render_detections_with_custom_style(image, results, model):
    """Render detections with custom confidence-based styling for YOLOv5"""
    # Convert PIL to OpenCV format if needed
    if isinstance(image, Image.Image):
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        image_cv = image.copy()
    
    # Get detections
    detections = results.xyxy[0]  # YOLOv5 format
    if len(detections) == 0:
        return image_cv
    
    class_names = results.names
    
    # Draw each detection
    for det in detections:
        x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
        confidence = float(det[4])
        class_id = int(det[5])
        class_name = str(class_names[class_id]).capitalize()
        
        # Get style for this confidence level
        style = get_confidence_style(confidence)
        
        # Draw rectangle with appropriate style
        draw_rectangle(image_cv, (x1, y1), (x2, y2), style['color'], style['thickness'], style['line_type'])
        
        # Always draw label with class name and confidence
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.39
        font_thickness = 1
        
        # Label shows class name and confidence value
        label = f"{class_name}: {confidence:.2f}"
        
        # Get text size
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        
        # Draw background rectangle for label with white background
        label_y = y1 - 10 if y1 > 30 else y2 + 25
        bg_rect_pt1 = (x1 - 2, label_y - text_size[1] - 6)
        bg_rect_pt2 = (x1 + text_size[0] + 6, label_y + 6)
        cv2.rectangle(image_cv, bg_rect_pt1, bg_rect_pt2, (255, 255, 255), -1)
        
        # Draw text with black color on white background
        cv2.putText(image_cv, label, (x1 + 2, label_y), font, font_scale, (0, 0, 0), font_thickness)
    
    return image_cv

def main():
    st.title("🔍 AI-Snaily - YOLOv5 Object Detection")
    st.markdown("YOLOv5 Object Detection Application")
    
    # Check if model exists
    model_path = "weights/v5.pt"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        st.info("Please ensure 'weights/v5.pt' exists in your project directory")
        st.stop()
    
    # Sidebar for configuration
    st.sidebar.header("Detection Configuration")
    
    # Detection parameters
    confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.45, 0.01)
    iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.01)
    
    # Source selection
    st.sidebar.header("Input Source")
    source_option = st.sidebar.selectbox(
        "Choose Input Source",
        ["Upload Image", "Upload Video", "Webcam"]
    )
    
    try:
        # Load model
        with st.spinner("Loading YOLOv5 model..."):
            model = load_yolov5_model(model_path)
        st.sidebar.success("✅ Model loaded successfully!")
        
        if source_option == "Upload Image":
            handle_image_detection(model, confidence, iou_threshold)
        elif source_option == "Upload Video":
            handle_video_detection(model, confidence, iou_threshold)
        elif source_option == "Webcam":
            handle_webcam_detection(model, confidence, iou_threshold)
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Make sure you have installed the required packages: pip install -r requirements.txt")

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
            
            # Run detection with timing
            with st.spinner("Running detection..."):
                start_time = time.time()
                results = model(image, conf=confidence, iou_threshold=iou_threshold)
                inference_time = time.time() - start_time
            
            # Display results with custom styling
            annotated_image = render_detections_with_custom_style(image, results, model)
            st.image(annotated_image, use_container_width=True, channels="BGR")
            
            # Display inference time
            st.info(f"⏱️ Inference Time: {inference_time:.3f} seconds")
            
            # Display detection statistics
            detections = results.xyxy[0]
            if len(detections) > 0:
                st.write(f"**Objects detected:** {len(detections)}")
                
                # Get class names and counts
                class_names = results.names
                class_counts = {}
                
                for det in detections:
                    class_id = int(det[5])
                    class_name = str(class_names[class_id]).capitalize()
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                # Display summary in nice format
                summary_text = ", ".join([f"{count} {class_name}" for class_name, count in sorted(class_counts.items())])
                st.write(f"**Found:** {summary_text}")
                
                # Detailed breakdown
                st.write("**Class Details:**")
                for class_name, count in sorted(class_counts.items()):
                    st.write(f"- {class_name}: {count}")
                
                # Display confidence distribution
                st.write("**Confidence Levels:**")
                confidences = detections[:, 4].numpy()
                low_conf = sum(1 for c in confidences if c < 0.35)
                mid_low_conf = sum(1 for c in confidences if 0.35 <= c < 0.60)
                mid_high_conf = sum(1 for c in confidences if 0.60 <= c < 0.85)
                high_conf = sum(1 for c in confidences if c >= 0.85)
                
                if low_conf > 0:
                    st.write(f"- VLow (<0.35): {low_conf}")
                if mid_low_conf > 0:
                    st.write(f"- Low (0.35-0.60): {mid_low_conf}")
                if mid_high_conf > 0:
                    st.write(f"- Mid (0.60-0.85): {mid_high_conf}")
                if high_conf > 0:
                    st.write(f"- High (>0.85): {high_conf}")
            else:
                st.write("**No objects detected**")

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
import torch

model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/v5.pt')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame, conf=0.45)
    annotated_frame = results.render()[0]
    
    cv2.imshow("YOLOv5 Detection", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
        '''
        
        st.code(webcam_code, language='python')

def process_video(model, video_path, confidence, iou_threshold):
    """Process video with YOLOv5 detection"""
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
    stats_placeholder = st.empty()
    
    frame_count = 0
    total_inference_time = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection with timing
        start_time = time.time()
        results = model(frame, conf=confidence, iou_threshold=iou_threshold)
        frame_inference_time = time.time() - start_time
        total_inference_time += frame_inference_time
        
        # Render with custom styling
        annotated_frame = render_detections_with_custom_style(frame, results, model)
        
        # Write frame
        out.write(annotated_frame)
        
        # Update progress
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        
        # Display stats
        avg_inference_time = total_inference_time / frame_count
        stats_placeholder.info(f"⏱️ Avg Inference Time: {avg_inference_time:.3f}s | Frame: {frame_count}/{total_frames}")
        
        # Display current frame (every 30th frame to avoid lag)
        if frame_count % 30 == 0:
            frame_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
    
    cap.release()
    out.release()
    
    st.success(f"Video processing complete! Total inference time: {total_inference_time:.2f}s")
    st.video("output_video.mp4")

if __name__ == "__main__":
    main()