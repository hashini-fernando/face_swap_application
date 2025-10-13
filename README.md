# Multi-Angle Face Swap App (Streamlit + InsightFace)

This project is a **Streamlit web app** that performs **AI-powered face swapping** while preserving **head pose and facial angles**.  
It uses **InsightFace**, **OpenCV**, and **ONNXRuntime** to detect, analyze, and swap faces from uploaded or live webcam images.

---

##  Features

- Face detection and embedding extraction using **InsightFace (buffalo_l)**  
- Realistic face swapping via **inswapper_128.onnx**  
- Angle-based matching (yaw/pitch/roll)  
- 3D head pose visualization  
- Live camera or image upload options  
- Quality metrics (PSNR, SSIM, Identity Similarity)  
- Download swapped results  
- Clean Streamlit dashboard UI  

---

##  Requirements

- Python 3.8 – 3.11
- pip

### Recommended: Create a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate      # on Windows
# or
source venv/bin/activate   # on macOS/Linux
```

### Create a requirements.txt file (or use the one included):
```bash
streamlit
opencv-python
numpy
Pillow
onnxruntime
insightface
scikit-image
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Running the App
```bash
streamlit run new.py

```

Project Flow


