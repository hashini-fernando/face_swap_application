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

### Project Flow

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/6b5912bb-9f01-4da1-8b9d-e2dad2693dea" />

### Face swapping appilication dashboard screen shots

<img width="1920" height="924" alt="image (1)" src="https://github.com/user-attachments/assets/120d5202-6170-4d8c-9181-d37c1ccd4d5a" />

<img width="1910" height="895" alt="image (3)" src="https://github.com/user-attachments/assets/6c7e142c-7633-492c-aba7-3cfa8e3c4bcd" />

<img width="1902" height="952" alt="image (2)" src="https://github.com/user-attachments/assets/ef3d34af-97c5-438d-8e57-8c3ea44d78f0" />

### Metrics

<img width="1920" height="881" alt="image" src="https://github.com/user-attachments/assets/bb3f40e1-b0e2-45f5-908a-83bf42da7977" />





