import streamlit as st
import cv2
import insightface
from insightface.app import FaceAnalysis
import numpy as np
from PIL import Image
import io
import math
import tempfile
import os

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
for key in ['face_image', 'swapped_image', 'target_image', 'intermediates', 'face_angle_warning']:
    if key not in st.session_state:
        st.session_state[key] = None if key != 'intermediates' else {}

# ---------------------------
# Utilities
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

# ---------------------------
# Face Pose Utilities (same as your code)
# ---------------------------
def estimate_face_pose(face_obj, image_shape):
    model_points = np.array([
        (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
    ])
    image_points = np.array([
        face_obj.kps[2],
        [(face_obj.kps[3][0] + face_obj.kps[4][0])/2, (face_obj.kps[3][1] + face_obj.kps[4][1])/2],
        face_obj.kps[0], face_obj.kps[1], face_obj.kps[3], face_obj.kps[4]
    ], dtype=np.float32)
    
    focal_length = image_shape[1]
    center = (image_shape[1]/2, image_shape[0]/2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4,1))
    success, rotation_vec, translation_vec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if success:
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pitch, yaw, roll = rotation_matrix_to_euler_angles(rotation_mat)
        return {'pitch': np.degrees(pitch), 'yaw': np.degrees(yaw), 'roll': np.degrees(roll),
                'rotation_vec': rotation_vec, 'translation_vec': translation_vec}
    return {'pitch': 0, 'yaw': 0, 'roll': 0}

def rotation_matrix_to_euler_angles(R):
    sy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
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
    angle_threshold = 45
    yaw_diff = abs(src_pose['yaw'] - dst_pose['yaw'])
    pitch_diff = abs(src_pose['pitch'] - dst_pose['pitch'])
    compatibility_score = max(0, 100 - (yaw_diff + pitch_diff))
    warnings = []
    if yaw_diff > angle_threshold:
        warnings.append(f"Large yaw difference: {yaw_diff:.1f}°")
    if pitch_diff > angle_threshold:
        warnings.append(f"Large pitch difference: {pitch_diff:.1f}°")
    return {'compatible': len(warnings)==0, 'score': compatibility_score, 'warnings': warnings, 'yaw_diff': yaw_diff, 'pitch_diff': pitch_diff}

def get_best_face_for_angle(faces, target_pose=None, method="angle_matching"):
    if not faces: return None
    if method=="angle_matching" and target_pose:
        best_face, best_score = None, -1
        for face in faces:
            pose = estimate_face_pose(face, (640, 640))
            compat = get_face_angle_compatibility(pose, target_pose)
            if compat['score'] > best_score:
                best_score = compat['score']
                best_face = face
        return best_face if best_face else faces[0]
    elif method=="most_frontal":
        best_face, best_score = None, float('inf')
        for face in faces:
            pose = estimate_face_pose(face, (640, 640))
            score = abs(pose['yaw']) + abs(pose['pitch'])
            if score < best_score:
                best_score = score
                best_face = face
        return best_face
    # default largest face
    areas = [(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]) for f in faces]
    return faces[np.argmax(areas)]



# ---------------------------
# Cached Models
# ---------------------------
@st.cache_resource(show_spinner=False)
def get_face_app():
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640,640))
    return app

@st.cache_resource(show_spinner=False)
def get_swapper():
    swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True)
    return swapper

# ---------------------------
# Video Utilities
# ---------------------------
def video_frames_generator(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret: break
        yield frame
    cap.release()

def save_video(frames, fps, size, output_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, size)
    for f in frames: out.write(f)
    out.release()
def perform_multi_angle_face_swap(face_image, target_image, use_angle_aware=True):
    """
    face_image: PIL.Image of source face
    target_image: PIL.Image of target image
    """
    app = get_face_app()
    swapper = get_swapper()

    face_cv2 = to_bgr(face_image)
    target_cv2 = to_bgr(target_image)

    face_faces = app.get(face_cv2)
    target_faces = app.get(target_cv2)

    if not face_faces:
        st.error("❌ No face detected in source image")
        return None, {}
    if not target_faces:
        st.error("❌ No face detected in target image")
        return None, {}

    # Use first face or angle-aware selection
    src_pose = estimate_face_pose(face_faces[0], face_cv2.shape)
    intermediates = {"Source_Pose": src_pose}

    result_image = target_cv2.copy()
    swapped_count = 0

    for t_face in target_faces:
        dst_pose = estimate_face_pose(t_face, target_cv2.shape)
        source_face = get_best_face_for_angle(face_faces, dst_pose, "angle_matching") if use_angle_aware else face_faces[0]
        try:
            result_image = swapper.get(result_image, t_face, source_face, paste_back=True)
            swapped_count += 1
        except Exception as e:
            print("Swap failed for one face:", e)
            continue

    intermediates["Total_Target_Faces"] = len(target_faces)
    intermediates["Swapped_Faces"] = swapped_count

    return to_pil(result_image), intermediates


def perform_multi_angle_face_swap_video(face_image, video_path, use_angle_aware=True):
    app = get_face_app()
    swapper = get_swapper()
    
    face_cv2 = to_bgr(face_image)
    face_faces = app.get(face_cv2)
    if not face_faces:
        st.error("❌ No face detected in source image")
        return None, {}
    
    src_pose = estimate_face_pose(face_faces[0], face_cv2.shape)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames_out = []
    total_faces, swapped_faces_count = 0, 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        target_faces = app.get(frame)
        total_faces += len(target_faces)
        frame_result = frame.copy()
        for t_face in target_faces:
            dst_pose = estimate_face_pose(t_face, frame.shape)
            source_face = get_best_face_for_angle(face_faces, dst_pose, "angle_matching") if use_angle_aware else face_faces[0]
            try: frame_result = swapper.get(frame_result, t_face, source_face, paste_back=True); swapped_faces_count += 1
            except: continue
        frames_out.append(frame_result)
    cap.release()
    
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    save_video(frames_out, fps, (w,h), temp_output)
    
    metrics = {"Total_Frames": len(frames_out), "Total_Faces": total_faces,
               "Swapped_Faces": swapped_faces_count, "Swap_Rate": swapped_faces_count/max(1,total_faces)*100}
    return temp_output, metrics

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    use_angle_aware = st.checkbox("Enable Angle-Aware Processing", value=True, help="Better results for non-frontal faces")

# ---------------------------
# Main UI
# ---------------------------
st.title("🔄 Multi-Angle Face Swap Application")
st.markdown("Swap faces in images or videos with multiple faces and various angles!")

col1, col2, col3 = st.columns(3)

# ---------------------------
# Column 1 - Face Image
# ---------------------------
with col1:
    st.markdown("### 📸 Face Photo")
    capture_method = st.radio("Choose input method:", ("Camera Capture","Upload Image"), key="face_method")
    if capture_method=="Camera Capture":
        camera_image = st.camera_input("Take a picture", key="camera_input")
        if camera_image: st.session_state.face_image = Image.open(camera_image)
    else:
        uploaded_face = st.file_uploader("Upload face image", type=['png','jpg','jpeg'], key="face_upload")
        if uploaded_face: st.session_state.face_image = Image.open(uploaded_face)
    if st.session_state.face_image: st.image(resize_image_for_display(st.session_state.face_image))
    
# ---------------------------
# Column 2 - Target Image / Video
# ---------------------------
with col2:
    st.markdown("### 🖼️ Target Image / Video")
    uploaded_target = st.file_uploader("Upload target image or video", type=['png','jpg','jpeg','mp4','avi','mov'], key="target_upload")
    if uploaded_target:
        file_ext = uploaded_target.name.split('.')[-1].lower()
        if file_ext in ['png','jpg','jpeg']:
            st.session_state.target_image = Image.open(uploaded_target)
            st.image(resize_image_for_display(st.session_state.target_image))
        else:
            temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}").name
            with open(temp_video_path, "wb") as f: f.write(uploaded_target.read())
            st.session_state.target_image = temp_video_path
            st.video(temp_video_path)

# ---------------------------
# Column 3 - Result
# ---------------------------
with col3:
    st.markdown("### ✨ Result")
    
    if st.button("🔄 Swap Faces", type="primary", use_container_width=True):
        if st.session_state.face_image and st.session_state.target_image:
            with st.spinner("Performing face swap..."):
                # Image swap
                if isinstance(st.session_state.target_image, Image.Image):
                    result, intermediates = perform_multi_angle_face_swap(
                        st.session_state.face_image, st.session_state.target_image, use_angle_aware
                    )
                    st.session_state.swapped_image = result
                    st.session_state.intermediates = intermediates
                    # Metrics for image
                    total_faces = intermediates.get('Source_Pose') and 1 or 0
                    st.markdown("### 📊 Performance Metrics")
                    st.metric("Total Faces Detected", total_faces)
                    st.metric("Swap Success", "✅" if result else "❌")
                    if result: st.image(resize_image_for_display(result))
                
                # Video swap
                else:
                    output_video_path, video_metrics = perform_multi_angle_face_swap_video(
                        st.session_state.face_image, st.session_state.target_image, use_angle_aware
                    )
                    st.markdown("### 📊 Performance Metrics")
                    st.metric("Total Frames", video_metrics["Total_Frames"])
                    st.metric("Total Faces Detected", video_metrics["Total_Faces"])
                    st.metric("Faces Swapped", video_metrics["Swapped_Faces"])
                    st.metric("Swap Success Rate (%)", f"{video_metrics['Swap_Rate']:.2f}")
                    st.video(output_video_path)
                    st.download_button(
                        "📥 Download Swapped Video",
                        data=open(output_video_path, 'rb').read(),
                        file_name="multi_angle_face_swap_video.mp4",
                        mime="video/mp4"
                    )
        else:
            st.error("❌ Please provide both face and target images or video")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>Multi-Angle Face Swap - Images & Videos, Multi-Face Compatible</div>", unsafe_allow_html=True)
