import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
import time

# Try to import plotly, but handle if it's not installed
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not installed. Charts will be disabled. Run: pip install plotly")

# Page configuration
st.set_page_config(
    page_title="Helmet Detector Pro",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .detection-badge {
        background: #FF4B4B;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'processed_frames' not in st.session_state:
    st.session_state.processed_frames = 0
if 'total_detections' not in st.session_state:
    st.session_state.total_detections = 0

# Title
st.markdown('<h1 class="main-header">🪖 Helmet Detection Pro</h1>', unsafe_allow_html=True)

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model loading
    @st.cache_resource
    def load_model():
        try:
            model_path = "best.pt"
            if not os.path.exists(model_path):
                st.error(f"Model file '{model_path}' not found!")
                return None
            return YOLO(model_path)
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None
    
    model = load_model()
    
    if model is None:
        st.warning("⚠️ Please make sure the model file 'best.pt' exists in the current directory")
        st.stop()
    
    # Confidence threshold
    confidence = st.slider(
        "Confidence Threshold",
        0.1, 1.0, 0.5, 0.05,
        help="Higher values reduce false positives but may miss detections"
    )
    
    # IOU threshold
    iou = st.slider(
        "IOU Threshold",
        0.1, 1.0, 0.45, 0.05,
        help="Intersection over Union threshold for non-maximum suppression"
    )
    
    # Detection classes
    st.subheader("🎯 Detection Classes")
    show_helmet = st.checkbox("Show Helmet", value=True)
    show_no_helmet = st.checkbox("Show No Helmet", value=True)
    show_head = st.checkbox("Show Head", value=True)
    
    # Display options
    st.subheader("🎨 Display Options")
    show_labels = st.checkbox("Show Labels", value=True)
    show_confidences = st.checkbox("Show Confidence Scores", value=True)
    
    # Input type selection with icons
    st.subheader("📁 Input Type")
    option = st.radio(
        "Choose Input Type",
        ["📷 Image", "🎥 Video"],
        horizontal=True
    )
    
    # Process button
    process_button = st.button("🚀 Process", use_container_width=True)

# Main content area
col1, col2, col3, col4 = st.columns(4)

# Detection statistics
with col1:
    st.metric(
        "🔄 Processed Frames",
        st.session_state.processed_frames,
        delta="+1" if st.session_state.processed_frames > 0 else None
    )

with col2:
    st.metric(
        "🎯 Total Detections",
        st.session_state.total_detections,
        delta="+1" if st.session_state.total_detections > 0 else None
    )

with col3:
    st.metric(
        "📊 Current Confidence",
        f"{confidence:.2f}",
        delta="Threshold"
    )

with col4:
    st.metric(
        "⏱️ Processing Time",
        "0.1s",
        delta="Per frame"
    )

# Helper function for enhanced bounding box rendering
def draw_enhanced_boxes(image, results, show_labels=True, show_confidences=True):
    img_copy = image.copy()
    detections = []
    
    if len(results[0].boxes) > 0:
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # Class names and colors
            class_names = ['Helmet', 'No Helmet', 'Head']
            class_colors = [(0, 255, 0), (0, 0, 255), (255, 165, 0)]
            
            # Filter classes
            if (cls == 0 and not show_helmet) or \
               (cls == 1 and not show_no_helmet) or \
               (cls == 2 and not show_head):
                continue
            
            # Draw rectangle
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), class_colors[cls], 2)
            
            # Draw label
            if show_labels:
                label = class_names[cls]
                if show_confidences:
                    label += f" {conf:.2f}"
                
                # Background for text
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img_copy, (x1, y1 - text_h - 10), (x1 + text_w, y1), class_colors[cls], -1)
                cv2.putText(img_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            detections.append({
                'class': class_names[cls],
                'confidence': conf,
                'bbox': [x1, y1, x2, y2]
            })
    
    return img_copy, detections

# ==========================================================
# IMAGE PROCESSING
# ==========================================================

if option == "📷 Image":
    uploaded_image = st.file_uploader(
        "📤 Upload Image",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Supported formats: JPG, JPEG, PNG, BMP, TIFF"
    )
    
    if uploaded_image is not None and process_button:
        try:
            with st.spinner("🔄 Processing image..."):
                # Load and prepare image
                image = Image.open(uploaded_image)
                if image.mode == "RGBA":
                    image = image.convert("RGB")
                
                # Display original
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📷 Original Image")
                    st.image(image, use_container_width=True)
                
                # Process image
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    image.save(tmp.name)
                    results = model.predict(
                        source=tmp.name,
                        conf=confidence,
                        iou=iou,
                        verbose=False
                    )
                    temp_path = tmp.name
                
                # Draw enhanced boxes
                img_array = np.array(image)
                annotated_img, detections = draw_enhanced_boxes(
                    img_array,
                    results,
                    show_labels,
                    show_confidences
                )
                
                # Update statistics
                st.session_state.processed_frames += 1
                st.session_state.total_detections += len(detections)
                
                # Display results
                with col2:
                    st.subheader("🎯 Detection Results")
                    st.image(annotated_img, use_container_width=True)
                
                # Detection statistics
                if detections:
                    df = pd.DataFrame(detections)
                    st.subheader("📊 Detection Statistics")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Detections", len(detections))
                    with col2:
                        helmets = len([d for d in detections if d['class'] == 'Helmet'])
                        st.metric("Helmets Found", helmets)
                    with col3:
                        no_helmets = len([d for d in detections if d['class'] == 'No Helmet'])
                        st.metric("No Helmet", no_helmets)
                    
                    # Confidence chart (only if plotly is available)
                    if PLOTLY_AVAILABLE:
                        fig = px.bar(
                            df,
                            x='class',
                            y='confidence',
                            color='class',
                            title='Detection Confidence Distribution',
                            labels={'confidence': 'Confidence Score', 'class': 'Class'}
                        )
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Fallback to simple table
                        st.dataframe(df[['class', 'confidence']], use_container_width=True)
                    
                    # Detection details
                    with st.expander("📋 Detection Details"):
                        st.dataframe(
                            df[['class', 'confidence']].style.background_gradient(subset=['confidence']),
                            use_container_width=True
                        )
                else:
                    st.warning("⚠️ No detections found. Try lowering the confidence threshold.")
                    
            # Clean up temp file
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
        
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)

# ==========================================================
# VIDEO PROCESSING
# ==========================================================

elif option == "🎥 Video":
    uploaded_video = st.file_uploader(
        "📤 Upload Video",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        help="Supported formats: MP4, AVI, MOV, MKV, WEBM"
    )
    
    if uploaded_video is not None and process_button:
        try:
            with st.spinner("🔄 Processing video..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(uploaded_video.read())
                    video_path = tmp.name
                    temp_video_path = tmp.name
                
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    st.error("❌ Could not open video file")
                    if os.path.exists(temp_video_path):
                        os.unlink(temp_video_path)
                    st.stop()
                
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Create output video
                output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                # Progress tracking
                progress_text = st.empty()
                progress_bar = st.progress(0)
                frame_placeholder = st.empty()
                
                # Video statistics
                st.subheader("📊 Video Statistics")
                stats_placeholder = st.empty()
                
                # Processing timer
                start_time = time.time()
                
                # Process frames
                processed_frames = 0
                total_detections = 0
                detection_times = []
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process frame
                    frame_start = time.time()
                    results = model(frame, conf=confidence, iou=iou, verbose=False)
                    frame_end = time.time()
                    detection_times.append(frame_end - frame_start)
                    
                    # Draw enhanced boxes
                    annotated_frame, detections = draw_enhanced_boxes(
                        frame,
                        results,
                        show_labels,
                        show_confidences
                    )
                    
                    # Write frame
                    writer.write(annotated_frame)
                    
                    # Update statistics
                    processed_frames += 1
                    total_detections += len(detections)
                    
                    # Update progress
                    progress = processed_frames / total_frames if total_frames > 0 else 0
                    progress_bar.progress(min(progress, 1.0))
                    progress_text.text(f"Processing frame {processed_frames}/{total_frames}")
                    
                    # Show frame
                    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, use_container_width=True)
                    
                    # Update statistics
                    if processed_frames % 10 == 0:
                        avg_time = np.mean(detection_times[-10:]) if detection_times else 0
                        if avg_time > 0:
                            stats_placeholder.metric("FPS", f"{1/avg_time:.1f}")
                        else:
                            stats_placeholder.metric("FPS", "N/A")
                
                # Release resources
                cap.release()
                writer.release()
                
                # Final statistics
                processing_time = time.time() - start_time
                st.session_state.processed_frames += processed_frames
                st.session_state.total_detections += total_detections
                
                # Display final statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Frames", processed_frames)
                with col2:
                    st.metric("Total Detections", total_detections)
                with col3:
                    st.metric("Avg FPS", f"{processed_frames/processing_time:.1f}" if processing_time > 0 else "0")
                with col4:
                    st.metric("Processing Time", f"{processing_time:.1f}s")
                
                st.success("✅ Video processing completed successfully!")
                
                # Display processed video
                st.subheader("🎥 Processed Video")
                st.video(output_path)
                
                # Download button
                with open(output_path, "rb") as f:
                    video_bytes = f.read()
                    st.download_button(
                        label="⬇️ Download Processed Video",
                        data=video_bytes,
                        file_name=f"helmet_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )
                
                # Cleanup
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)
                if os.path.exists(output_path):
                    os.unlink(output_path)
        
        except Exception as e:
            st.error(f"❌ Error processing video: {str(e)}")
            if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
            if 'output_path' in locals() and os.path.exists(output_path):
                os.unlink(output_path)

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### 📊 Summary")
    st.write(f"Total processed frames: {st.session_state.processed_frames}")
    st.write(f"Total detections: {st.session_state.total_detections}")
    
with col2:
    st.markdown("### 📈 Detection Rate")
    if st.session_state.processed_frames > 0:
        detection_rate = st.session_state.total_detections / st.session_state.processed_frames
        st.metric("Avg Detections/Frame", f"{detection_rate:.2f}")

with col3:
    st.markdown("### 🔄 Session Info")
    st.write(f"Session started: {datetime.now().strftime('%H:%M:%S')}")
    if st.button("🔄 Reset Statistics", use_container_width=True):
        st.session_state.detection_history = []
        st.session_state.processed_frames = 0
        st.session_state.total_detections = 0
        st.rerun()

# Instructions
with st.expander("ℹ️ How to use"):
    st.markdown("""
    1. **Upload your media**: Choose an image or video file
    2. **Adjust settings**: Fine-tune confidence threshold and display options
    3. **Process**: Click the 'Process' button to run detection
    4. **View results**: See detections with bounding boxes and statistics
    5. **Download**: Save processed videos for later use
    
    **Tips**:
    - Lower confidence threshold for more detections (may include false positives)
    - Higher confidence threshold for more accurate detections (may miss some)
    - For video processing, lower resolution videos will process faster
    """)
