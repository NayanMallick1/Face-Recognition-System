# Face-Recognition-System
Face Recognition System built with Python, OpenCV, and MTCNN for accurate face detection, embedding generation, and real-time recognition. Includes a training script to capture and store facial embeddings and a recognition module for fast webcam-based identification, ideal for security and authentication.
# Face Recognition System

[![Python](https://img.shields.io/badge/python-3.8+-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green)](https://opencv.org/)
[![DeepFace](https://img.shields.io/badge/DeepFace-0.0.79-brightgreen)](https://github.com/serengil/deepface)

---

## Description
A high-accuracy **Face Recognition System** using multi-algorithm detection (MTCNN, RetinaFace, Haar Cascade) with DeepFace's ArcFace embeddings. This system provides robust face validation, real-time webcam recognition, and automated attendance tracking with Excel reporting.

Key capabilities:
- 🎯 Face detection with automatic algorithm selection
- 📊 Attendance management with time-based deduplication
- 🔍 Incremental training for new faces
- 🖼️ Image, video, and webcam input support
- 📁 Automatic logging and reporting

---

## Features
- **Triple Detection System**: MTCNN, RetinaFace, and Haar Cascade with automatic fallback
- **Advanced Validation**: Aspect ratio, skin tone, texture, and symmetry checks
- **Real-time Recognition**: Webcam processing with frame optimization
- **Attendance Tracking**: Automatic Excel reports with timestamps
- **Incremental Learning**: Add new faces without full retraining
- **Multi-source Input**: Process images, videos, and live camera feeds
- **Duplicate Prevention**: Configurable time-based attendance filtering

---
# Real-time Recognition Demo
<img width="914" height="523" alt="image" src="https://github.com/user-attachments/assets/15755600-72a7-478d-834e-76b0c16590d2" />

### Terminal Output During Recognition:
**[ RetinaFace Detection ]**  
Similarity with suchandrika: 0.27  
Similarity with swastika: 0.32  
Similarity with adrija_sengupta: 0.43  
...  
Similarity with adrija_halder: 0.28  
Similarity with indranil: 0.41  
Similarity with nayan: 0.77  
...  

**[ MTCNN Detection ]**  
Detected 1 face using mtcnn  
Similarity with adrija_halder: 0.30  
Similarity with indranil: 0.44  
Similarity with nayan: 0.74  
...  

**[ Haar Cascade Detection ]**  
Detected 2 faces using haar  
Similarity with adrija_halder: 0.27  
Similarity with indranil: 0.40  
Similarity with nayan: 0.78  
...  
Similarity with adrija_halder: 0.99  
Similarity with indranil: 0.99  
Similarity with nayan: 1.00  
...  

**[ Final Attendance ]**  
Marked for **nayan** (Accuracy ~76%)  


**Attendance:** Marked for **nayan** (Accuracy: 75.95%)

---
### Sample Output (Recognition Summary)
- **RetinaFace** → top match **nayan** (0.77)  
- **MTCNN** → top match **nayan** (0.74)  
- **Haar** → top match **nayan** (0.78 → 1.00)  
- **Final Attendance** → **nayan** (~76% accuracy)  

---

## Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/NayanMallick1/Face-Recognition-System.git
cd Face-Recognition-System
```
---
### 2️⃣ Create virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```
---
### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 4️⃣ Install system dependencies
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y cmake libopenblas-dev liblapack-dev libjpeg-dev

# Windows (additional dependencies)
pip install opencv-python-headless
```
---
## Usage
---
### 🧠 Training the model
```bash
python train_face_recognizer.py
```
- Creates `face_embeddings.pkl` and `processed_images.json`
- Generates `candidates.xlsx` registry
- Add new images to `faceData/PersonName/` and rerun
---
### 👁️ Running recognition
```bash
python face_recognition_main.py
```
1. **Image Recognition**  
   - Process single image file

2. **Webcam Recognition**  
   - Real-time detection (press 'q' to exit)

3. **Video File Recognition**  
   - Process video files
---
### 📁 Dataset structure
```bash
faceData/
├── Person1/
│   ├── image1.jpg
│   ├── image2.png
├── Person2/
│   ├── photo1.jpeg
```
---
### File Structure
```bash
├── faceData/                  # Training dataset
├── detected_faces/            # Webcam capture output
├── attendance_2023-08-13.xlsx # Auto-generated attendance
├── candidates.xlsx            # Candidate registry
├── face_embeddings.pkl        # Trained embeddings
├── processed_images.json      # Training metadata
├── train_face_recognizer.py   # Training script
├── face_recognition_main.py   # Main recognition module
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```
---
### Configuration
Adjust parameters in `face_recognition_main.py:`
```bash
# Detection thresholds (Line 84)
if face['confidence'] > 0.01  # MTCNN confidence

# Validation parameters (validate_face function)
aspect_ratio = (0.6, 1.6)    # Line 35
min_face_size = 30            # Line 38
texture_threshold = 15        # Line 48
skin_ratio_threshold = 0.15   # Line 57

# Recognition threshold (Line 119)
if max_score > 0.65           # Similarity threshold

# Attendance interval (Line 157)
if time_diff.total_seconds() < 1800  # 30 minutes
```
---
### Troubleshooting
```bash
| Issue                        | Solution                                            |
|------------------------------|-----------------------------------------------------|
| "Face embeddings file not found" | Run `train_face_recognizer.py` first            |
| High CPU usage               | Increase modulo in webcam loop (Line 269)           |
| Webcam not detected          | Try different camera indices (Line 250)             |
| Poor low-light detection     | Adjust HSV ranges in `validate_face()`              |
| False positives              | Increase similarity threshold (Line 119)            |
```
---
#### Note: For optimal performance, use 5+ high-quality images per person with consistent lighting conditions.
---

## License

MIT License  

Copyright (c) 2025 Nayan Mallick  

Permission is hereby granted, free of charge, to any person obtaining a copy  
of this software and associated documentation files (the "Software"), to deal  
in the Software without restriction, including without limitation the rights  
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell  
copies of the Software, and to permit persons to whom the Software is  
furnished to do so, subject to the following conditions:  

The above copyright notice and this permission notice shall be included in all  
copies or substantial portions of the Software.  

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,  
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE  
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER  
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  
SOFTWARE.  

---
