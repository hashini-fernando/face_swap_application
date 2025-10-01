import streamlit as st
import cv2
import insightface
from insightface.app import FaceAnalysis
import numpy as np
from PIL import Image
import io
import math

# ---------------------------
# Page configuration + styles
# ---------------------------
st.set_page_config(page_title="Multi-Angle Face Swap App", layout="wide")
st.markdown("""
<style>
    .main { padding: 0rem 0rem; }
    .block-container { padding-top: 2rem; padding-bottom: 0rem; padding-left: 5rem; padding-right: 5rem; }
    div[data-testid="column"] {
        background-color: #f9f9f9; border: 2px solid #ddd; border-radius: 10px; padding: 20px; min-height: 500px;
    }
    .stImage { max-width: 100%; margin: 20px auto; display: block; }
    h3 { text-align: center; color: #333; margin-bottom: 20px; }
    .stButton > button { width: 100%; background-color: #4CAF50; color: white; font-weight: bold; margin-top: 10px; }
    .stDownloadButton > button { width: 100%; background-color: #008CBA; color: white; font-weight: bold; }
    .warning-box { background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px; padding: 10px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Session state
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

# ---------------------------
# Enhanced Utilities for Multi-Angle Faces
# ---------------------------
def resize_image_for_display(image, max_width=280, max_height=350):
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
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def to_pil(img_bgr):
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

# -----------------------------------------
# Face Angle Detection and Analysis
# -----------------------------------------
def estimate_face_pose(face_obj, image_shape):
    """Estimate face pose (pitch, yaw, roll) from landmarks"""
    # 3D model points for reference
    model_points = np.array([
        (0.0, 0.0, 0.0),        # Nose tip
        (0.0, -330.0, -65.0),   # Chin
        (-225.0, 170.0, -135.0), # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corner
        (-150.0, -150.0, -125.0), # Left mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ])
    
    # 2D image points from detected landmarks
    image_points = np.array([
        face_obj.kps[2],        # Nose tip (index 2 in InsightFace 5-point)
        [(face_obj.kps[3][0] + face_obj.kps[4][0])/2, (face_obj.kps[3][1] + face_obj.kps[4][1])/2],  # Approximate chin
        face_obj.kps[0],        # Left eye left corner
        face_obj.kps[1],        # Right eye right corner
        face_obj.kps[3],        # Left mouth corner
        face_obj.kps[4]         # Right mouth corner
    ], dtype=np.float32)
    
    # Camera internals approximation
    focal_length = image_shape[1]
    center = (image_shape[1]/2, image_shape[0]/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    
    dist_coeffs = np.zeros((4,1))  # Assuming no lens distortion
    
    # Solve PnP
    success, rotation_vec, translation_vec = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, 
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if success:
        # Convert rotation vector to matrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        
        # Extract Euler angles
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

def get_face_angle_compatibility(src_pose, dst_pose):
    """Check if face angles are compatible for swapping"""
    angle_threshold = 45  
    
    yaw_diff = abs(src_pose['yaw'] - dst_pose['yaw'])
    pitch_diff = abs(src_pose['pitch'] - dst_pose['pitch'])
    
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
        # Find face with most similar angle to target
        best_face = None
        best_score = -1
        
        for face in faces:
            face_pose = estimate_face_pose(face, (640, 640))  # Default shape
            if target_pose:
                compatibility = get_face_angle_compatibility(face_pose, target_pose)
                if compatibility['score'] > best_score:
                    best_score = compatibility['score']
                    best_face = face
        return best_face if best_face else faces[0]
    
    elif method == "most_frontal":
        # Find most frontal face (smallest absolute angles)
        best_face = None
        best_angle_score = float('inf')
        
        for face in faces:
            pose = estimate_face_pose(face, (640, 640))
            angle_score = abs(pose['yaw']) + abs(pose['pitch'])
            if angle_score < best_angle_score:
                best_angle_score = angle_score
                best_face = face
        return best_face
    
    # Default: largest face
    areas = [(face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]) for face in faces]
    return faces[np.argmax(areas)]

# -----------------------------------------
# Enhanced Face Swapping for Multi-Angle
# -----------------------------------------
@st.cache_resource(show_spinner=False)
def get_face_app():
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

@st.cache_resource(show_spinner=False)
def get_swapper():
    swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True)
    return swapper

def enhanced_similarity_transform_for_angle(src_kps, dst_kps, src_pose, dst_pose):
    """Enhanced alignment that considers face angles"""
    src = np.asarray(src_kps, dtype=np.float32)
    dst = np.asarray(dst_kps, dtype=np.float32)
    
    # Adjust weights based on angle compatibility
    yaw_diff = abs(src_pose['yaw'] - dst_pose['yaw'])
    
    if yaw_diff > 30:  # Large angle difference
        # Give more weight to stable features (eyes, nose)
        weights = np.array([1.5, 1.5, 2.0, 1.0, 1.0])
    else:
        weights = np.array([1.2, 1.2, 1.5, 1.0, 1.0])
    
    M, _ = cv2.estimateAffinePartial2D(src, dst, weights=weights, 
                                     method=cv2.RANSAC, ransacReprojThreshold=5.0)
    return M

def adaptive_face_mask(img_shape, face_obj, pose):
    """Create mask that adapts to face pose"""
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    kps = face_obj.kps.astype(int)
    x1, y1, x2, y2 = face_obj.bbox.astype(int)
    
    # Adjust mask based on yaw angle
    yaw = abs(pose['yaw'])
    
    if yaw < 15:  # Frontal face
        # Use elliptical mask
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        rx = int((x2 - x1) / 2 * 1.1)
        ry = int((y2 - y1) / 2 * 1.0)
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    
    elif yaw < 45:  # Moderate angle
        # Use convex hull with expansion
        hull = cv2.convexHull(kps)
        expanded_hull = expand_contour(hull, scale=1.15)
        cv2.fillConvexPoly(mask, expanded_hull, 255)
    
    else:  # Profile face
        # Use landmark-based polygon
        profile_poly = create_profile_polygon(kps, pose['yaw'])
        cv2.fillPoly(mask, [profile_poly], 255)
    
    # Feather edges
    feather_size = max(5, int(min(h, w) * 0.01))
    mask = cv2.GaussianBlur(mask, (feather_size*2+1, feather_size*2+1), 0)
    
    return mask

def expand_contour(contour, scale=1.1):
    """Expand contour around center"""
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0
    
    expanded = []
    for point in contour:
        x, y = point[0]
        new_x = int(cx + (x - cx) * scale)
        new_y = int(cy + (y - cy) * scale)
        expanded.append([new_x, new_y])
    
    return np.array(expanded, dtype=np.int32)

def create_profile_polygon(kps, yaw):
    """Create polygon for profile faces"""
    # For simplicity, use convex hull with extra points for profile
    hull = cv2.convexHull(kps)
    
    # Add extra area on the visible side for profile faces
    if yaw > 0:  # Facing left, add area to the right
        extra_x = int(kps[:, 0].max() + (kps[:, 0].max() - kps[:, 0].min()) * 0.2)
    else:  # Facing right, add area to the left
        extra_x = int(kps[:, 0].min() - (kps[:, 0].max() - kps[:, 0].min()) * 0.2)
    
    # Add extra points to hull
    extra_points = []
    for point in hull:
        x, y = point[0]
        if (yaw > 0 and x == kps[:, 0].max()) or (yaw < 0 and x == kps[:, 0].min()):
            extra_points.append([extra_x, y])
    
    if extra_points:
        hull = np.vstack([hull, np.array(extra_points)])
        hull = cv2.convexHull(hull)
    
    return hull

def draw_pose_visualization(img_bgr, face_obj, pose):
    """Draw pose estimation visualization"""
    vis = img_bgr.copy()
    
    # Draw coordinate axes
    size = 50
    axis_points = np.float32([[size, 0, 0], [0, -size, 0], [0, 0, -size]]).reshape(-1, 3)
    
    # Project 3D points to 2D
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
    
    # Draw axes
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR: X, Y, Z
    for i, color in enumerate(colors):
        cv2.line(vis, tuple(nose_tip.astype(int)), tuple(img_points[i][0]), color, 2)
    
    # Add angle text
    text = f"Yaw: {pose['yaw']:.1f}°, Pitch: {pose['pitch']:.1f}°"
    cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return vis

def perform_multi_angle_face_swap(face_image, target_image, use_angle_aware=True):
    """Enhanced face swap that handles various angles"""
    try:
        app = get_face_app()
        swapper = get_swapper()

        face_cv2 = to_bgr(face_image)
        target_cv2 = to_bgr(target_image)

        face_faces = app.get(face_cv2)
        target_faces = app.get(target_cv2)

        if not face_faces:
            st.error("❌ No face detected in the face image")
            return None, {}
        if not target_faces:
            st.error("❌ No face detected in the target image")
            return None, {}

        # Analyze poses
        src_poses = [estimate_face_pose(face, face_cv2.shape) for face in face_faces]
        dst_poses = [estimate_face_pose(face, target_cv2.shape) for face in target_faces]

        # Select best source face considering angles
        if use_angle_aware and dst_poses:
            # Try to match angles with the first target face
            source_face = get_best_face_for_angle(face_faces, dst_poses[0], "angle_matching")
            
            # Check compatibility
            src_pose = estimate_face_pose(source_face, face_cv2.shape)
            compatibility = get_face_angle_compatibility(src_pose, dst_poses[0])
            
            if compatibility['warnings']:
                st.session_state.face_angle_warning = f"⚠️ Angle differences detected: {', '.join(compatibility['warnings'])}. Results may be suboptimal."
            else:
                st.session_state.face_angle_warning = "✅ Good angle match!"
                
        else:
            source_face = get_best_face_for_angle(face_faces, method="most_frontal")
            st.session_state.face_angle_warning = ""

        # Perform swap
        result = target_cv2.copy()
        for i, (target_face, dst_pose) in enumerate(zip(target_faces, dst_poses)):
            result = swapper.get(result, target_face, source_face, paste_back=True)

        # Create intermediates for visualization
        intermediates = {}
        if use_angle_aware:
            intermediates["Source_Pose"] = draw_pose_visualization(
                to_bgr(face_image), source_face, src_pose
            )
            intermediates["Target_Pose"] = draw_pose_visualization(
                target_cv2, target_faces[0], dst_poses[0]
            )

        return to_pil(result), intermediates

    except Exception as e:
        st.error(f"❌ Error during face swap: {str(e)}")
        return None, {}

# ---------------------------
# UI – Enhanced for Multi-Angle
# ---------------------------
with st.sidebar:
   
    
    use_angle_aware = st.checkbox("Enable Angle-Aware Processing", value=True, 
                                 help="Better results for non-frontal faces")
 


# ---------------------------
# Main UI
# ---------------------------
st.title("🔄 Multi-Angle Face Swap Application")
st.markdown("Swap faces in images with various angles - not just frontal views!")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📸 Face Photo")
    capture_method = st.radio("Choose input method:", ("Camera Capture", "Upload Image"), key="face_method")
    if capture_method == "Camera Capture":
        camera_image = st.camera_input("Take a picture", key="camera_input")
        if camera_image is not None:
            st.session_state.face_image = Image.open(camera_image)
            st.image(resize_image_for_display(st.session_state.face_image))
    else:
        uploaded_face = st.file_uploader("Upload face image", type=['png', 'jpg', 'jpeg'], key="face_upload")
        if uploaded_face is not None:
            st.session_state.face_image = Image.open(uploaded_face)
            st.image(resize_image_for_display(st.session_state.face_image))
    
    if st.session_state.face_image is not None:
        # Quick face angle analysis
        face_cv2 = to_bgr(st.session_state.face_image)
        app = get_face_app()
        faces = app.get(face_cv2)
        if faces:
            pose = estimate_face_pose(faces[0], face_cv2.shape)
            angle_info = f"Estimated: Yaw {pose['yaw']:.1f}°, Pitch {pose['pitch']:.1f}°"
            if abs(pose['yaw']) > 30 or abs(pose['pitch']) > 30:
                st.warning(f"⚠️ Non-frontal face detected: {angle_info}")
            else:
                st.success(f"✅ Frontal face: {angle_info}")

with col2:
    st.markdown("### 🖼️ Target Image")
    uploaded_target = st.file_uploader("Upload target image", type=['png', 'jpg', 'jpeg'], key="target_upload")
    if uploaded_target is not None:
        st.session_state.target_image = Image.open(uploaded_target)
        st.image(resize_image_for_display(st.session_state.target_image))
        
        # Quick target angle analysis
        target_cv2 = to_bgr(st.session_state.target_image)
        app = get_face_app()
        faces = app.get(target_cv2)
        if faces:
            pose = estimate_face_pose(faces[0], target_cv2.shape)
            angle_info = f"Estimated: Yaw {pose['yaw']:.1f}°, Pitch {pose['pitch']:.1f}°"
            if abs(pose['yaw']) > 30 or abs(pose['pitch']) > 30:
                st.warning(f"⚠️ Non-frontal face: {angle_info}")
            else:
                st.success(f"✅ Frontal face: {angle_info}")

with col3:
    st.markdown("### ✨ Result")
    
    if st.button("🔄 Swap Faces", type="primary", use_container_width=True):
        if st.session_state.face_image is not None and st.session_state.target_image is not None:
            with st.spinner("🔍 Performing angle-aware face swap..."):
                result, intermediates = perform_multi_angle_face_swap(
                    st.session_state.face_image, 
                    st.session_state.target_image,
                    use_angle_aware=use_angle_aware
                )
                
                if result is not None:
                    st.session_state.swapped_image = result
                    st.session_state.intermediates = intermediates
                    
                    # Show angle compatibility warning
                    if st.session_state.face_angle_warning:
                        if "⚠️" in st.session_state.face_angle_warning:
                            st.warning(st.session_state.face_angle_warning)
                        else:
                            st.success(st.session_state.face_angle_warning)
                    
                    st.success("✅ Face swap completed!")
        else:
            st.error("❌ Please provide both face and target images")

    if st.session_state.swapped_image is not None:
        st.image(resize_image_for_display(st.session_state.swapped_image))
        
        buf = io.BytesIO()
        st.session_state.swapped_image.save(buf, format='PNG')
        st.download_button(
            label="📥 Download Result",
            data=buf.getvalue(),
            file_name="multi_angle_face_swap.png",
            mime="image/png",
            use_container_width=True
        )
    else:
        st.info("🎭 Click 'Swap Faces' to see the result")

# ---------------------------
# Pose Visualization
# ---------------------------
if (st.session_state.intermediates and 
    "Source_Pose" in st.session_state.intermediates and 
    use_angle_aware):

    st.markdown("## 🎯 Pose Analysis")
    st.info("Visualization of face poses and angles")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Source Face Pose**")
        st.image(to_pil(st.session_state.intermediates["Source_Pose"]), 
                use_container_width=True)
    
    with col2:
        st.markdown("**Target Face Pose**")
        st.image(to_pil(st.session_state.intermediates["Target_Pose"]), 
                use_container_width=True)

# ---------------------------
# Tips for Non-Frontal Faces
# ---------------------------
with st.expander("💡 Tips for Non-Frontal Faces"):
    st.markdown("""
    **For best results with angled faces:**
    
    ### 🎯 **Angle Matching**
    - Use source and target faces with similar angles
    - Avoid swapping between frontal and profile faces
    - Yaw differences > 45° will have poor results
    
    ### 📐 **Pose Considerations**
    - **Frontal (0-15°)**: Best results
    - **Moderate (15-45°)**: Good results with angle-aware processing
    - **Profile (>45°)**: Limited functionality, may need manual adjustment
    
    ### 🌟 **Lighting & Quality**
    - Ensure good lighting on the visible side of the face
    - Higher resolution images work better
    - Avoid heavy shadows on angled faces
    
    ### 🔧 **Technical Limitations**
    - Extreme profile views may not swap correctly
    - Faces looking up/down excessively may have artifacts
    - The system works best when both faces have similar orientations
    """)

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Multi-Angle Face Swap - Handles frontal, angled, and profile faces"
    "</div>",
    unsafe_allow_html=True
)