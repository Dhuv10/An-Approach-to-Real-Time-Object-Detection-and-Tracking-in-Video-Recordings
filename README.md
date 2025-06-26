# 🎥 Real-Time Object Detection and Tracking

A Python-based project implementing real-time object detection and tracking in video recordings using TensorFlow and OpenCV. Ideal for applications like video surveillance, autonomous vehicles, and augmented reality systems.

---

## 📌 Project Overview

This project combines:
- Object detection using pre-trained models (YOLOv3/SSD or TensorFlow detection API)
- Real-time object tracking (e.g., centroid tracker, SORT, or OpenCV trackers)
- Visualization with bounding boxes and object IDs
- Application to real-world video footage and COCO 2017 dataset

---

## 🧰 Technologies Used

- Python
- OpenCV
- TensorFlow / TensorFlow Object Detection API
- COCO 2017 Dataset
- Numpy, imutils

---

## 🧪 Features

- Real-time detection on video input
- Tracking across frames with persistent IDs
- Supports webcam or video file input
- Frame-wise accuracy validation with COCO annotations

---

## 📂 Project Structure

```plaintext
detection/         # Scripts for loading model and detecting objects
tracking/          # Scripts for tracking logic
utils/             # Helper functions for I/O, frame capture
results/           # Screenshots and sample outputs
requirements.txt   # Python dependencies
