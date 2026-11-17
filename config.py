import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'gradcam_xai_face_visualizer_secret_key_2026')
    
    # Storage Folders
    STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
    UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, 'uploads')
    SAMPLES_FOLDER = os.path.join(STATIC_FOLDER, 'samples')
    IMG_FOLDER = os.path.join(STATIC_FOLDER, 'img')
    
    # Database
    DB_PATH = os.path.join(BASE_DIR, 'deepfake_gradcam.db')
    
    # File Restrictions
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # Maksimum 16MB
    
    # Model & Grad-CAM Settings
    MODEL_INPUT_SIZE = (224, 224)
    TARGET_LAYER_NAME = 'layer4'  # Layer konvolusi terakhir ResNet
    CASCADE_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
    YUNET_PATH = os.path.join(BASE_DIR, 'face_detection_yunet_2023mar.onnx')
    
    # Project Metadata (Fokus Penuh pada Visualisasi Grad-CAM & XAI)
    PROJECT_TITLE = "Visualisasi Grad-CAM Citra Wajah Berbasis Explainable AI (XAI)"
    SYSTEM_NAME = "FaceGradCAM"
    SUBTITLE = "Sistem Transparansi AI & Visualisasi Peta Perhatian Konvolusi (Grad-CAM)"
