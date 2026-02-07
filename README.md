# Real-Time Object Detection using OpenCV

This project demonstrates a simple yet effective real-time object detection system using classical computer vision techniques.  
The application captures live video from a webcam, processes each frame, detects objects using contour analysis, and displays bounding boxes along with object counts.

## 🔍 Features
- Live webcam video processing
- Adaptive thresholding for lighting robustness
- Noise reduction using Gaussian blur
- Contour-based object detection
- Real-time bounding box visualization
- Object count display

## 🛠️ Technologies Used
- Python
- OpenCV
- NumPy

## 🚀 How It Works
1. Capture video frames from the webcam
2. Convert frames to grayscale
3. Reduce noise using Gaussian blur
4. Apply adaptive thresholding
5. Detect contours as objects
6. Draw bounding boxes and count objects in real time

## ▶️ How to Run

```bash
pip install -r requirements.txt
python src/main.py
