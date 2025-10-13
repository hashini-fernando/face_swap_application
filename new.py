import streamlit as st
import cv2
import insightface
from insightface.app import FaceAnalysis
import numpy as np
from PIL import Image
import io
import math
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from math import log10
from sklearn.metrics.pairwise import cosine_similarity
# ---------------------------
# Page configuration + Enhanced styles
# ---------------------------
st.set_page_config(
    page_title="🎭 Multi-Angle Face Swap App", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { padding: 0rem 0rem; }
    .block-container { 
        padding-top: 2rem; 
        padding-bottom: 2rem; 
        padding-left: 3rem; 
        padding-right: 3rem; 
        max-width: 1400px;
    }
    
    div[data-testid="column"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border: 2px solid #e1e8ed;
        border-radius: 15px;
        padding: 25px;
        min-height: 500px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    div[data-testid="column"]:hover {
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    
    .stImage { 
        max-width: 100%; 
        margin: 20px auto; 
        display: block;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    h1 {
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        text-align: center;
        margin-bottom: 10px;
    }
    
    h3 { 
        text-align: center; 
        color: #2c3e50;
        margin-bottom: 20px;
        font-weight: 600;
    }
    
    .stButton > button { 
        width: 100%; 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; 
        font-weight: bold; 
        margin-top: 10px;
        border: none;
        border-radius: 10px;
        padding: 12px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    .stDownloadButton > button { 
        width: 100%; 
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white; 
        font-weight: bold;
        border: none;
        border-radius: 10px;
        padding: 12px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(17, 153, 142, 0.3);
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(17, 153, 142, 0.4);
    }
    
    .success-box {
    background: linear-gradient(135deg, #e8f9e9 0%, #c7f5d9 100%); /* Softer green */
    color: #111;                          /* Black text */
    border-radius: 8px;                   /* Smooth corners */
    padding: 15px 20px;                   /* Balanced spacing */
    margin: 12px 0;
    border-left: 6px solid #28a745;       /* Strong success green */
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08); /* Subtle depth */
    font-size: 15px;
    line-height: 1.5;
    animation: slideIn 0.4s ease-out;
}

    
   .warning-box {
    background-color: #fff8e1;        
    color: #333;                      
    border-radius: 8px;             
    padding: 15px 20px;               
    margin: 12px 0;
    border-left: 6px solid #ff9800;   
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08); 
    font-size: 15px;
    line-height: 1.5;
    animation: slideIn 0.4s ease-out;
}
    
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .sidebar-content {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Session state initialization
# ---------------------------
if 'face_image' not in st.session_state:
    st.session_state.face_image = None
if 'swapped_image' not in st.session_state:
    st.session_state.swapped_image = None
if 'target_image' not in st.session_state:
    st.session_state.target_image = None
if 'intermediates' not in st.session_state:
    st.session_state.intermediates = {}
if 'face_angle_warning' not in st.session_state:
    st.session_state.face_angle_warning = ""
if 'swap_history' not in st.session_state:
    st.session_state.swap_history = []
if 'processing_time' not in st.session_state:
    st.session_state.processing_time = 0
if 'quality_settings' not in st.session_state:
    st.session_state.quality_settings = {'det_size': 640, 'blend_ratio': 1.0}

# ---------------------------
# Enhanced Utilities
# ---------------------------
def resize_image_for_display(image, max_width=280, max_height=350):
    """Resize image for display with aspect ratio preservation"""
    width, height = image.size
    aspect_ratio = width / height
    if width > max_width or height > max_height:
        if width / max_width > height / max_height:
            new_width = max_width
            new_height = int(max_width / aspect_ratio)
        else:
            new_height = max_height
            new_width = int(max_height * aspect_ratio)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image

def to_bgr(img_pil):
    """Convert PIL image to BGR"""
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def to_pil(img_bgr):
    """Convert BGR image to PIL"""
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

def apply_image_enhancement(image, brightness=1.0, contrast=1.0):
    """Apply brightness and contrast adjustments"""
    img_array = np.array(image).astype(np.float32)
    img_array = img_array * contrast
    img_array = img_array + (brightness - 1.0) * 128
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)

# ---------------------------
# Face Angle Detection and Analysis
# ---------------------------
def estimate_face_pose(face_obj, image_shape):
    """Estimate face pose (pitch, yaw, roll) from landmarks"""
    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ])
    
    image_points = np.array([
        face_obj.kps[2],
        [(face_obj.kps[3][0] + face_obj.kps[4][0])/2, (face_obj.kps[3][1] + face_obj.kps[4][1])/2],
        face_obj.kps[0],
        face_obj.kps[1],
        face_obj.kps[3],
        face_obj.kps[4]
    ], dtype=np.float32)
    
    focal_length = image_shape[1]
    center = (image_shape[1]/2, image_shape[0]/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    
    dist_coeffs = np.zeros((4,1))
    
    success, rotation_vec, translation_vec = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, 
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if success:
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pitch, yaw, roll = rotation_matrix_to_euler_angles(rotation_mat)
        
        return {
            'pitch': np.degrees(pitch),
            'yaw': np.degrees(yaw),
            'roll': np.degrees(roll),
            'rotation_vec': rotation_vec,
            'translation_vec': translation_vec
        }
    
    return {'pitch': 0, 'yaw': 0, 'roll': 0}

def rotation_matrix_to_euler_angles(R):
    """Convert rotation matrix to Euler angles"""
    sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6
    
    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    
    return np.array([x, y, z])

def angular_diff(a, b):
    """Compute the shortest angular difference between two angles"""
    diff = abs(a - b) % 360
    return diff if diff <= 180 else 360 - diff

def get_face_angle_compatibility(src_pose, dst_pose):
    """Check if face angles are compatible for swapping"""
    angle_threshold = 45
    yaw_diff = angular_diff(src_pose['yaw'], dst_pose['yaw'])
    pitch_diff = angular_diff(src_pose['pitch'], dst_pose['pitch'])
    compatibility_score = max(0, 100 - (yaw_diff + pitch_diff))
    
    warnings = []
    if yaw_diff > angle_threshold:
        warnings.append(f"Large yaw difference: {yaw_diff:.1f}°")
    if pitch_diff > angle_threshold:
        warnings.append(f"Large pitch difference: {pitch_diff:.1f}°")
    
    return {
        'compatible': len(warnings) == 0,
        'score': compatibility_score,
        'warnings': warnings,
        'yaw_diff': yaw_diff,
        'pitch_diff': pitch_diff
    }

def get_best_face_for_angle(faces, target_pose=None, method="angle_matching"):
    """Select best face considering angles"""
    if not faces:
        return None
    
    if method == "angle_matching" and target_pose:
        best_face = None
        best_score = -1
        
        for face in faces:
            face_pose = estimate_face_pose(face, (640, 640))
            if target_pose:
                compatibility = get_face_angle_compatibility(face_pose, target_pose)
                if compatibility['score'] > best_score:
                    best_score = compatibility['score']
                    best_face = face
        return best_face if best_face else faces[0]
    
    elif method == "most_frontal":
        best_face = None
        best_angle_score = float('inf')
        
        for face in faces:
            pose = estimate_face_pose(face, (640, 640))
            angle_score = abs(pose['yaw']) + abs(pose['pitch'])
            if angle_score < best_angle_score:
                best_angle_score = angle_score
                best_face = face
        return best_face
    
    areas = [(face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]) for face in faces]
    return faces[np.argmax(areas)]

def draw_pose_visualization(img_bgr, face_obj, pose):
    """Draw pose estimation visualization"""
    vis = img_bgr.copy()
    
    size = 50
    axis_points = np.float32([[size, 0, 0], [0, -size, 0], [0, 0, -size]]).reshape(-1, 3)
    
    focal_length = img_bgr.shape[1]
    center = (img_bgr.shape[1]/2, img_bgr.shape[0]/2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype=np.float32)
    
    dist_coeffs = np.zeros((4,1))
    
    nose_tip = face_obj.kps[2].astype(np.float32)
    rotation_vec = pose.get('rotation_vec', np.zeros((3,1)))
    translation_vec = pose.get('translation_vec', np.array([[0], [0], [1000]]))
    
    img_points, _ = cv2.projectPoints(axis_points, rotation_vec, translation_vec, 
                                    camera_matrix, dist_coeffs)
    img_points = img_points.astype(int)
    
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for i, color in enumerate(colors):
        cv2.line(vis, tuple(nose_tip.astype(int)), tuple(img_points[i][0]), color, 3)
    
    text = f"Yaw: {pose['yaw']:.1f}° | Pitch: {pose['pitch']:.1f}° | Roll: {pose['roll']:.1f}°"
    cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    return vis

# ---------------------------
# Enhanced Face Swapping
# ---------------------------
@st.cache_resource(show_spinner=False)
def get_face_app(det_size=640):
    """Initialize face analysis app"""
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(det_size, det_size))
    return app

@st.cache_resource(show_spinner=False)
def get_swapper():
    """Initialize face swapper model"""
    swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True)
    return swapper

def perform_multi_angle_face_swap(face_image, target_image, use_angle_aware=True, blend_ratio=1.0):
    """Enhanced face swap with angle awareness and blending"""
    try:
        start_time = datetime.now()
        
        det_size = st.session_state.quality_settings['det_size']
        app = get_face_app(det_size)
        swapper = get_swapper()

        face_cv2 = to_bgr(face_image)
        target_cv2 = to_bgr(target_image)

        face_faces = app.get(face_cv2)
        target_faces = app.get(target_cv2)

        if not face_faces:
            st.error("❌ No face detected in the face image")
            return None, {}, 0
        if not target_faces:
            st.error("❌ No face detected in the target image")
            return None, {}, 0

        src_poses = [estimate_face_pose(face, face_cv2.shape) for face in face_faces]
        dst_poses = [estimate_face_pose(face, target_cv2.shape) for face in target_faces]

        if use_angle_aware and dst_poses:
            source_face = get_best_face_for_angle(face_faces, dst_poses[0], "angle_matching")
            src_pose = estimate_face_pose(source_face, face_cv2.shape)
            compatibility = get_face_angle_compatibility(src_pose, dst_poses[0])
            
            if compatibility['warnings']:
                st.session_state.face_angle_warning = f"⚠️ {', '.join(compatibility['warnings'])}"
            else:
                st.session_state.face_angle_warning = " Excellent angle match!"
        else:
            source_face = get_best_face_for_angle(face_faces, method="most_frontal")
            st.session_state.face_angle_warning = ""

        result = target_cv2.copy()
        for target_face in target_faces:
            result = swapper.get(result, target_face, source_face, paste_back=True)

        if blend_ratio < 1.0:
            result = cv2.addWeighted(target_cv2, 1 - blend_ratio, result, blend_ratio, 0)

        intermediates = {}
        if use_angle_aware:
            intermediates["Source_Pose"] = draw_pose_visualization(face_cv2, source_face, src_pose)
            intermediates["Target_Pose"] = draw_pose_visualization(target_cv2, target_faces[0], dst_poses[0])

        processing_time = (datetime.now() - start_time).total_seconds()
        return to_pil(result), intermediates, processing_time

    except Exception as e:
        st.error(f"❌ Error during face swap: {str(e)}")
        return None, {}, 0

# ---------------------------
# Metrics Computation
# ---------------------------
def compute_face_detection_rate(app, face_image, target_image):
    """Return detection rate for both images"""
    face_cv2 = to_bgr(face_image)
    target_cv2 = to_bgr(target_image)
    
    face_detected = len(app.get(face_cv2)) > 0
    target_detected = len(app.get(target_cv2)) > 0
    
    return (face_detected + target_detected) / 2 * 100

def compute_pose_metrics(face_obj, target_obj):
    """Return pose metrics and compatibility"""
    src_pose = estimate_face_pose(face_obj, (640, 640))
    dst_pose = estimate_face_pose(target_obj, (640, 640))
    compatibility = get_face_angle_compatibility(src_pose, dst_pose)
    
    return {
        'src_yaw': src_pose['yaw'],
        'src_pitch': src_pose['pitch'],
        'src_roll': src_pose['roll'],
        'dst_yaw': dst_pose['yaw'],
        'dst_pitch': dst_pose['pitch'],
        'dst_roll': dst_pose['roll'],
        'angle_compatibility_score': compatibility['score'],
        'warnings': compatibility['warnings']
    }
def compute_psnr(img1, img2):
    """Compute PSNR between two images"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * log10(max_pixel / np.sqrt(mse))

def compute_ssim(img1, img2):
    """Compute SSIM between two RGB images"""
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(img1_gray, img2_gray, full=True)
    return score

def compute_identity_similarity(app, img1, img2):
    """Compute cosine similarity between face embeddings"""
    faces1 = app.get(img1)
    faces2 = app.get(img2)
    if not faces1 or not faces2:
        return None
    emb1 = faces1[0].normed_embedding.reshape(1, -1)
    emb2 = faces2[0].normed_embedding.reshape(1, -1)
    return cosine_similarity(emb1, emb2)[0][0]
# ---------------------------
# Sidebar Configuration
# ---------------------------
with st.sidebar:
    st.markdown("##  Configuration")
    
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.markdown("###  Processing Options")
    use_angle_aware = st.checkbox("Enable Angle-Aware Processing", value=True, 
                                 help="Better results for non-frontal faces")
    
    st.markdown("###  Quality Settings")
    quality_preset = st.select_slider(
        "Quality Preset",
        options=["Fast", "Balanced", "High Quality"],
        value="Balanced"
    )
    
    if quality_preset == "Fast":
        st.session_state.quality_settings['det_size'] = 320
    elif quality_preset == "Balanced":
        st.session_state.quality_settings['det_size'] = 640
    else:
        st.session_state.quality_settings['det_size'] = 1024
    
    blend_ratio = st.slider("Blend Intensity", 0.5, 1.0, 1.0, 0.05,
                           help="Lower values = more subtle blend")
    st.session_state.quality_settings['blend_ratio'] = blend_ratio
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.markdown("### 📊 Statistics")
    if st.session_state.swap_history:
        st.metric("Total Swaps", len(st.session_state.swap_history))
        avg_time = sum(st.session_state.swap_history) / len(st.session_state.swap_history)
        st.metric("Avg Processing Time", f"{avg_time:.2f}s")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("🗑️ Clear All", use_container_width=True):
        st.session_state.face_image = None
        st.session_state.target_image = None
        st.session_state.swapped_image = None
        st.session_state.intermediates = {}
        st.rerun()

# ---------------------------
# Main UI
# ---------------------------
st.title("Multi-Angle Face Swap Studio")
st.markdown(
    "<p style='text-align: center; color: #666; font-size: 1.1em;'>"
    "face swapping for frontal, angled, and profile views"
    "</p>", 
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Source Face")
    capture_method = st.radio("Input Method:", ("Upload Image", "Camera Capture"), key="face_method")
    
    if capture_method == "Camera Capture":
        camera_image = st.camera_input("Take a picture", key="camera_input")
        if camera_image is not None:
            st.session_state.face_image = Image.open(camera_image)
    else:
        uploaded_face = st.file_uploader("Upload face image", type=['png', 'jpg', 'jpeg'], key="face_upload")
        if uploaded_face is not None:
            st.session_state.face_image = Image.open(uploaded_face)
    
    if st.session_state.face_image is not None:
        st.image(resize_image_for_display(st.session_state.face_image))
        
        face_cv2 = to_bgr(st.session_state.face_image)
        app = get_face_app()
        faces = app.get(face_cv2)
        
        if faces:
            pose = estimate_face_pose(faces[0], face_cv2.shape)
            angle_info = f"Yaw: {pose['yaw']:.1f}° | Pitch: {pose['pitch']:.1f}°"
            
            if abs(pose['yaw']) > 30 or abs(pose['pitch']) > 30:
                st.markdown(f'<div class="warning-box">Non-frontal: {angle_info}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="success-box"> Frontal: {angle_info}</div>', unsafe_allow_html=True)

with col2:
    st.markdown("###  Target Image")
    uploaded_target = st.file_uploader("Upload target image", type=['png', 'jpg', 'jpeg'], key="target_upload")
    
    if uploaded_target is not None:
        st.session_state.target_image = Image.open(uploaded_target)
        st.image(resize_image_for_display(st.session_state.target_image))
        
        target_cv2 = to_bgr(st.session_state.target_image)
        app = get_face_app()
        faces = app.get(target_cv2)
        
        if faces:
            pose = estimate_face_pose(faces[0], target_cv2.shape)
            angle_info = f"Yaw: {pose['yaw']:.1f}° | Pitch: {pose['pitch']:.1f}°"
            
            if abs(pose['yaw']) > 30 or abs(pose['pitch']) > 30:
                st.markdown(f'<div class="warning-box">⚠️ Non-frontal: {angle_info}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="success-box">✅ Frontal: {angle_info}</div>', unsafe_allow_html=True)

with col3:
    st.markdown("###  Result")
    
    if st.button(" Swap Faces", type="primary", use_container_width=True):
        if st.session_state.face_image is not None and st.session_state.target_image is not None:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔍 Detecting faces...")
            progress_bar.progress(25)
            
            status_text.text("🎯 Analyzing angles...")
            progress_bar.progress(50)
            
            status_text.text(" Swapping faces...")
            result, intermediates, proc_time = perform_multi_angle_face_swap(
                st.session_state.face_image, 
                st.session_state.target_image,
                use_angle_aware=use_angle_aware,
                blend_ratio=blend_ratio
            )
            progress_bar.progress(100)
            
            if result is not None:
                st.session_state.swapped_image = result
                st.session_state.intermediates = intermediates
                st.session_state.processing_time = proc_time
                st.session_state.swap_history.append(proc_time)
                
                status_text.empty()
                progress_bar.empty()
                
                if st.session_state.face_angle_warning:
                    if "⚠️" in st.session_state.face_angle_warning:
                        st.markdown(f'<div class="warning-box">{st.session_state.face_angle_warning}</div>', 
                                  unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="success-box">{st.session_state.face_angle_warning}</div>', 
                                  unsafe_allow_html=True)
                
                
        else:
            st.warning("⚠️ Please upload both images first!")

    if st.session_state.swapped_image is not None:
        st.image(resize_image_for_display(st.session_state.swapped_image))
        
        st.metric("⏱️ Processing Time", f"{st.session_state.processing_time:.2f}s")
        
        buf = io.BytesIO()
        st.session_state.swapped_image.save(buf, format='PNG')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="📥 Download Result",
            data=buf.getvalue(),
            file_name=f"face_swap_{timestamp}.png",
            mime="image/png",
            use_container_width=True
        )
    else:
        st.info(" Upload images and click 'Swap Faces' ")

# ---------------------------
# Metrics Dashboard
# ---------------------------
if st.session_state.swapped_image is not None:
    st.markdown("---")
    st.markdown("##  Quality Metrics Dashboard")
    
    metric_cols = st.columns(4)
    
    app = get_face_app()
    face_cv2 = to_bgr(st.session_state.face_image)
    target_cv2 = to_bgr(st.session_state.target_image)
    
    face_faces = app.get(face_cv2)
    target_faces = app.get(target_cv2)
    
    with metric_cols[0]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        detection_rate = compute_face_detection_rate(app, st.session_state.face_image, st.session_state.swapped_image)
        st.metric("Detection Rate", f"{detection_rate:.0f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metric_cols[1]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if face_faces and target_faces:
            pose_metrics = compute_pose_metrics(face_faces[0], target_faces[0])
            st.metric(" Angle Match", f"{pose_metrics['angle_compatibility_score']:.0f}%")
        else:
            st.metric(" Angle Match", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metric_cols[2]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        face_count = len(face_faces) if face_faces else 0
        target_count = len(target_faces) if target_faces else 0
        st.metric("Faces Detected", f"{face_count} → {target_count}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metric_cols[3]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        quality_score = 85 if detection_rate == 100 else 70
        st.metric(" Quality Score", f"{quality_score}%")
        st.markdown('</div>', unsafe_allow_html=True)
    with metric_cols[2]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        swapped_cv2 = to_bgr(st.session_state.swapped_image)
        ssim_val = compute_ssim(target_cv2, swapped_cv2)
        st.metric("SSIM", f"{ssim_val:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with metric_cols[3]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        psnr_val = compute_psnr(target_cv2, swapped_cv2)
        st.metric("PSNR", f"{psnr_val:.2f} dB")
        st.markdown('</div>', unsafe_allow_html=True)

    # Identity similarity (add below the above metrics)
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    identity_sim = compute_identity_similarity(app, face_cv2, swapped_cv2)
    if identity_sim is not None:
        st.metric("Identity Similarity", f"{identity_sim:.3f}")
    else:
        st.metric("Identity Similarity", "N/A")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Detailed angle analysis
    if face_faces and target_faces and pose_metrics:
        st.markdown("### Detailed Angle Analysis")
        
        angle_col1, angle_col2 = st.columns(2)
        
        with angle_col1:
            st.markdown("**Source Face Angles**")
            angle_data = {
                "Yaw (Left/Right)": f"{pose_metrics['src_yaw']:.1f}°",
                "Pitch (Up/Down)": f"{pose_metrics['src_pitch']:.1f}°",
                "Roll (Tilt)": f"{pose_metrics['src_roll']:.1f}°"
            }
            for key, value in angle_data.items():
                st.text(f"{key}: {value}")
        
        with angle_col2:
            st.markdown("**Target Face Angles**")
            angle_data = {
                "Yaw (Left/Right)": f"{pose_metrics['dst_yaw']:.1f}°",
                "Pitch (Up/Down)": f"{pose_metrics['dst_pitch']:.1f}°",
                "Roll (Tilt)": f"{pose_metrics['dst_roll']:.1f}°"
            }
            for key, value in angle_data.items():
                st.text(f"{key}: {value}")
        
        if pose_metrics['warnings']:
            st.warning("⚠️ Angle Warnings: " + ", ".join(pose_metrics['warnings']))

# ---------------------------
# Pose Visualization
# ---------------------------
if (st.session_state.intermediates and 
    "Source_Pose" in st.session_state.intermediates and 
    use_angle_aware):
    
    st.markdown("---")
    st.markdown("##  3D Pose Visualization")
    st.info("Visual representation of face orientation in 3D space")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        st.markdown("**Source Face Orientation**")
        st.image(to_pil(st.session_state.intermediates["Source_Pose"]), 
                use_container_width=True)
        st.caption("Red: X-axis | Green: Y-axis | Blue: Z-axis")
    
    with viz_col2:
        st.markdown("**Target Face Orientation**")
        st.image(to_pil(st.session_state.intermediates["Target_Pose"]), 
                use_container_width=True)
        st.caption("Red: X-axis | Green: Y-axis | Blue: Z-axis")

# ---------------------------
# Comparison View
# ---------------------------
if st.session_state.face_image and st.session_state.target_image and st.session_state.swapped_image:
    st.markdown("---")
    st.markdown("## 🔍 Before & After Comparison")
    
    comparison_cols = st.columns(3)
    
    with comparison_cols[0]:
        st.markdown("**Original Source**")
        st.image(st.session_state.face_image, use_container_width=True)
    
    with comparison_cols[1]:
        st.markdown("**Original Target**")
        st.image(st.session_state.target_image, use_container_width=True)
    
    with comparison_cols[2]:
        st.markdown("**Final Result**")
        st.image(st.session_state.swapped_image, use_container_width=True)



# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding: 20px;'>
        <p style='color: #666; font-size: 0.9em;'>
             <strong>Multi-Angle Face Swap Studio</strong> <br>
            Computer Vision and Image Processing<br>
            <em>Powered by InsightFace & OpenCV</em>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)