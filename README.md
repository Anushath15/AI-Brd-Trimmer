# AI Brd Trimmer

> AI-powered Augmented Reality Beard Trimming Assistant using Computer Vision and Artificial Intelligence.

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-5.x-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Landmarker-orange.svg)
![Status](https://img.shields.io/badge/Status-Under%20Development-yellow)
![License](https://img.shields.io/badge/License-MIT-blue)

---

# Overview

AI Brd Trimmer is an intelligent beard grooming assistant that combines Artificial Intelligence, Computer Vision, and Augmented Reality to help users trim their beard accurately.

Instead of relying on guesswork, the application analyzes facial features, detects the beard region, recommends suitable beard styles, and overlays a real-time trimming guide directly onto the user's live camera feed.

The goal is to provide a safer, more precise, and beginner-friendly beard trimming experience.

---

# Features

## Camera Input

- Upload a beard photo
- Live webcam support
- Camera calibration
- Face alignment verification

---

## AI Face Analysis

- Face Landmark Detection
- Jawline Detection
- Chin Detection
- Beard Segmentation
- Beard Density Analysis
- Growth Pattern Analysis

---

## AI Beard Recommendation

- AI-based beard style recommendation
- Face-shape analysis
- Style catalogue
- Custom beard designer

---

## AR Trimming Assistant

- Live beard overlay
- Target beard boundary generation
- Trim zone generation
- Dynamic face alignment
- Real-time overlay rendering

---

## Smart Guidance

Visual Guidance

- Keep Zone (Green)
- Blend Zone (Yellow)
- Trim Zone (Red)

Voice Guidance

- Trim instructions
- Boundary alerts
- Progress notifications

---

## Progress Tracking

- Live trimming progress
- Touch-up detection
- Final comparison
- Before & After preview

---

# Workflow

```
Start
│
├── Choose Input
│      ├── Upload Image
│      └── Live Camera
│
├── Camera Calibration
│      ├── Face Centered
│      ├── Lighting Check
│      └── Distance Check
│
├── Face Landmark Detection
│
├── Beard Segmentation
│
├── Growth & Density Analysis
│
├── AI Style Recommendation
│
├── Choose Beard Style
│
├── Generate Beard Boundary
│
├── Generate Trimming Zones
│
├── Live AR Overlay
│
├── Visual + Voice Guidance
│
├── Progress Tracking
│
├── Touch-up Detection
│
├── Before / After Preview
│
└── Finish
```

---

# Project Structure

```
AI-Brd-Trimmer/

├── config/
├── src/
│   ├── app/
│   ├── capture/
│   ├── comparison/
│   ├── detection/
│   ├── guidance/
│   ├── overlay/
│   ├── progress/
│   ├── styles/
│   ├── tracking/
│   ├── utils/
│   └── zones/
│
├── assets/
├── data/
├── docs/
├── tests/
├── README.md
└── requirements.txt
```

---

# Technology Stack

| Category | Technology |
|-----------|------------|
| Language | Python |
| Computer Vision | OpenCV |
| Face Detection | MediaPipe |
| Image Processing | NumPy |
| Visualization | OpenCV Drawing APIs |
| Voice Guidance | pyttsx3 |
| Configuration | YAML |
| Testing | PyTest |

---

# Installation

Clone the repository

```bash
git clone https://github.com/Anushath15/AI-Brd-Trimmer-.git
```

Enter the project

```bash
cd AI-Brd-Trimmer
```

Create Virtual Environment

```bash
python -m venv .venv
```

Activate Virtual Environment

Windows

```bash
.venv\Scripts\activate
```

Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run

```bash
python src/main.py
```

---

# Development Roadmap

## Version 1

- Camera Calibration
- Face Landmark Detection
- Beard Segmentation
- Density Analysis
- AI Recommendation
- Live Overlay
- Visual Guidance
- Voice Guidance
- Progress Tracking
- Before & After Preview

---

## Version 2

- Blade Tracking
- Hair Length Estimation
- Automatic Trim Progress
- Personalized Beard Profiles
- Cloud Synchronization
- Mobile Application
- 3D Face Reconstruction
- AI Learning from User Preferences

---

# Future Improvements

- YOLO Beard Segmentation
- Deep Learning Style Recommendation
- Reinforcement Learning Personalization
- Face Mesh Stabilization
- Mobile Deployment
- Offline AI Model
- Multi-language Voice Guidance
- Real-time Beard Simulation

---

# Testing

Run all tests

```bash
pytest
```

---

# Project Status

Current Status

> Under Active Development

Completed

- Project Architecture
- Repository Structure
- Development Roadmap

In Progress

- Camera Module
- Face Detection
- Beard Segmentation

Upcoming

- AR Overlay
- AI Recommendation
- Progress Tracking

---

# Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push the branch
5. Open a Pull Request

---

# License

This project is licensed under the MIT License.

---

# Author

**Anushath S**

Computer Science Engineering Student

Artificial Intelligence & Machine Learning Enthusiast

GitHub

https://github.com/Anushath15

---

# Vision

To make professional beard grooming accessible to everyone using Artificial Intelligence, Computer Vision, and Augmented Reality.

---

⭐ If you like this project, consider giving it a star.