import os
import time
import math
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageOps

from config import Config
from gradcam import (
    PyTorchGradCAM,
    refine_targeted_gradcam,
    compute_precision_regions,
    apply_colormap_on_cam,
    blend_cam_overlay
)

# ==============================================================================
# 1. ARSITEKTUR MODEL NEURAL NETWORK & GRAD-CAM ENGINE
# ==============================================================================

class FacialFeatureCNN(nn.Module):
    """
    Arsitektur Deep Learning ResNet-18 untuk Analisis Fitur Wajah dan
    Ekstraksi Peta Aktivasi Grad-CAM pada Layer Konvolusi Terakhir (layer4).
    """
    def __init__(self, pretrained=True):
        super(FacialFeatureCNN, self).__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base_resnet = models.resnet18(weights=weights)
        
        self.conv1 = base_resnet.conv1
        self.bn1 = base_resnet.bn1
        self.relu = base_resnet.relu
        self.maxpool = base_resnet.maxpool
        
        self.layer1 = base_resnet.layer1
        self.layer2 = base_resnet.layer2
        self.layer3 = base_resnet.layer3
        self.layer4 = base_resnet.layer4  # Target Layer Grad-CAM (512 Channels)
        
        self.avgpool = base_resnet.avgpool
        
        in_features = base_resnet.fc.in_features
        self.feature_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.feature_head(x)
        return logits


_MODEL_INSTANCE = None
_GRADCAM_INSTANCE = None
_YUNET_DETECTOR = None

def get_model_and_gradcam():
    global _MODEL_INSTANCE, _GRADCAM_INSTANCE
    if _MODEL_INSTANCE is None:
        model = FacialFeatureCNN(pretrained=True)
        model.eval()
        target_layer = model.layer4[-1].conv2
        gradcam = PyTorchGradCAM(model, target_layer)
        _MODEL_INSTANCE = model
        _GRADCAM_INSTANCE = gradcam
    return _MODEL_INSTANCE, _GRADCAM_INSTANCE

def get_yunet_detector():
    global _YUNET_DETECTOR
    if _YUNET_DETECTOR is None and hasattr(cv2, 'FaceDetectorYN') and os.path.exists(Config.YUNET_PATH):
        try:
            _YUNET_DETECTOR = cv2.FaceDetectorYN.create(Config.YUNET_PATH, '', (300, 300))
        except Exception:
            _YUNET_DETECTOR = None
    return _YUNET_DETECTOR


# ==============================================================================
# 2. DETEKSI WAJAH & EKSTRAKSI LANDMARK FITUR ANATOMIS
# ==============================================================================

preprocess_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def detect_and_crop_face(image_path):
    """
    Mendeteksi wajah dan 5 landmark kunci (kedua mata, hidung, dan kedua sudut mulut)
    menggunakan model YuNet. Melakukan framing kepala proporsional berdimensi 224x224.
    Mendukung auto-orientasi EXIF untuk kamera HP iOS (iPhone) & Android.
    """
    try:
        # Menangani orientasi EXIF (foto vertikal dari kamera iPhone / Android)
        pil_img = Image.open(image_path)
        pil_img = ImageOps.exif_transpose(pil_img)
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        # Tulis ulang gambar yang sudah terorientasi tegak lurus
        pil_img.save(image_path, format="JPEG", quality=95)
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        img_bgr = cv2.imread(image_path)
        
    if img_bgr is None:
        raise ValueError(f"Tidak dapat membaca gambar dari {image_path}")
        
    orig_h, orig_w = img_bgr.shape[:2]
    detector = get_yunet_detector()
    
    face_box = None
    landmarks_224 = None
    has_detected_face = False
    
    if detector is not None:
        try:
            detector.setInputSize((orig_w, orig_h))
            _, faces = detector.detect(img_bgr)
            if faces is not None and len(faces) > 0:
                best_face = faces[0]
                fx, fy, fw, fh = [int(v) for v in best_face[:4]]
                
                # Framing kepala: tambahkan margin 36%
                cx = fx + fw // 2
                cy = fy + fh // 2
                crop_size = int(max(fw, fh) * 1.36)
                
                x1 = max(0, cx - crop_size // 2)
                y1 = max(0, cy - crop_size // 2)
                x2 = min(orig_w, x1 + crop_size)
                y2 = min(orig_h, y1 + crop_size)
                
                actual_w = x2 - x1
                actual_h = y2 - y1
                side = max(actual_w, actual_h)
                x2 = min(orig_w, x1 + side)
                y2 = min(orig_h, y1 + side)
                actual_w = x2 - x1
                actual_h = y2 - y1
                
                face_crop_bgr = img_bgr[y1:y2, x1:x2]
                face_box = (x1, y1, actual_w, actual_h)
                has_detected_face = True
                
                def to_224(lx, ly):
                    px = int((lx - x1) / actual_w * 224)
                    py = int((ly - y1) / actual_h * 224)
                    return (max(0, min(223, px)), max(0, min(223, py)))
                
                landmarks_224 = {
                    'right_eye': to_224(best_face[4], best_face[5]),
                    'left_eye': to_224(best_face[6], best_face[7]),
                    'nose': to_224(best_face[8], best_face[9]),
                    'right_mouth': to_224(best_face[10], best_face[11]),
                    'left_mouth': to_224(best_face[12], best_face[13])
                }
        except Exception as e:
            print(f"[YuNet Detection Exception] {e}")

    # Fallback jika model deteksi belum menemukan wajah
    if not has_detected_face:
        crop_size = min(orig_w, orig_h)
        left = (orig_w - crop_size) // 2
        top = max(0, int((orig_h - crop_size) * 0.20))
        right = left + crop_size
        bottom = min(orig_h, top + crop_size)
        face_crop_bgr = img_bgr[top:bottom, left:right]
        face_box = (left, top, crop_size, crop_size)
        landmarks_224 = {
            'right_eye': (78, 86),
            'left_eye': (146, 86),
            'nose': (112, 120),
            'right_mouth': (84, 154),
            'left_mouth': (140, 154)
        }

    face_224_bgr = cv2.resize(face_crop_bgr, (224, 224), interpolation=cv2.INTER_AREA)
    return img_bgr, face_224_bgr, face_box, has_detected_face, landmarks_224


# ==============================================================================
# 3. ANALISIS SALIENSI & KONSENTRASI FITUR GRAD-CAM
# ==============================================================================

def analyze_facial_energy(face_224_bgr):
    """
    Menghitung energi spasial dan residual frekuensi untuk pembobotan heatmap Grad-CAM.
    """
    gray = cv2.cvtColor(face_224_bgr, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = 20 * np.log(np.abs(fshift) + 1e-8)
    
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_std = float(np.std(cv2.absdiff(gray, blur)))
    return {'fft_energy': float(np.mean(mag)), 'laplacian_var': lap, 'noise_std': noise_std}


# ==============================================================================
# 4. PIPELINE VISUALISASI GRAD-CAM & TANDA SASARAN FITUR WAJAH
# ==============================================================================

def run_deepfake_gradcam_pipeline(image_path, base_output_name):
    """
    Menghasilkan Visualisasi Peta Aktivasi Grad-CAM murni dan tanda sasaran fitur wajah.
    Fokus utama: Peta Perhatian Konvolusi, Heatmap Jet, dan Tanda Organ Wajah.
    """
    start_time = time.time()
    
    # 1. Deteksi Wajah & Koordinat Landmark
    img_bgr, face_224_bgr, face_box, has_face, landmarks_224 = detect_and_crop_face(image_path)
    
    # 2. Input Tensor PyTorch
    face_rgb = cv2.cvtColor(face_224_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_rgb)
    input_tensor = preprocess_transform(pil_img).unsqueeze(0)
    
    # 3. Model & PyTorch Grad-CAM Hook
    model, gradcam = get_model_and_gradcam()
    cues = analyze_facial_energy(face_224_bgr)
    
    # 4. Ekstraksi Grad-CAM dari Layer Konvolusi Terakhir (layer4)
    raw_cam, logits, grad_energy = gradcam.generate_cam(input_tensor, target_class=1)
    
    # 5. Eksekusi Grad-CAM Presisi Terfokus pada Landmark Wajah
    # Selalu menghasilkan heatmap aktif yang menonjolkan fitur-fitur wajah
    targeted_cam = refine_targeted_gradcam(
        raw_cam,
        face_224_bgr,
        landmarks=landmarks_224,
        verdict='DEEPFAKE',  # Mode aktif untuk visualisasi penuh
        target_width=224,
        target_height=224
    )
    
    # 6. Hitung Skor Aktivasi Per Wilayah & Kotak Sasaran Wajah
    region_data = compute_precision_regions(targeted_cam, landmarks=landmarks_224, is_deepfake=True)
    scores = region_data['scores']
    peak_coord = region_data['peak_coord']
    peak_name = region_data['peak_name']
    peak_intensity = region_data['peak_intensity']
    target_boxes = region_data['target_boxes']

    # 7. Render Heatmap Visual (Colormap JET) & Overlay
    heatmap_colored, heatmap_gray, _ = apply_colormap_on_cam(targeted_cam, 224, 224, cv2.COLORMAP_JET)
    overlay_bgr = blend_cam_overlay(face_224_bgr, heatmap_colored, targeted_cam, alpha_base=0.65)

    # 8. Susun Rincian Fokus Perhatian per Organ Wajah
    eye_score = scores['eyes']
    jaw_score = scores['jawline']
    mouth_score = scores['mouth']
    skin_score = scores['skin']

    regions = [
        {
            'id': 'eyes',
            'name': 'Area Mata & Alis',
            'icon': 'eye',
            'score': eye_score,
            'is_edited': eye_score > 60,
            'status': 'Aktivasi Tinggi' if eye_score > 70 else ('Aktivasi Sedang' if eye_score > 50 else 'Aktivasi Normal'),
            'severity': 'Tinggi' if eye_score > 75 else ('Sedang' if eye_score > 50 else 'Rendah'),
            'explanation': (
                f'Grad-CAM mendeteksi konsentrasi gradien {eye_score}% pada pupil dan kelopak mata. Fitur mata memberikan kontribusi signifikan terhadap pola representasi spasial model konvolusi.'
            )
        },
        {
            'id': 'jawline',
            'name': 'Garis Rahang & Batas Wajah',
            'icon': 'crop',
            'score': jaw_score,
            'is_edited': jaw_score > 60,
            'status': 'Aktivasi Tinggi' if jaw_score > 70 else ('Aktivasi Sedang' if jaw_score > 50 else 'Aktivasi Normal'),
            'severity': 'Tinggi' if jaw_score > 75 else ('Sedang' if jaw_score > 50 else 'Rendah'),
            'explanation': (
                f'Grad-CAM menyoroti kontur tepi rahang dengan bobot {jaw_score}%. Menunjukkan sensitivitas filter konvolusi terhadap garis transisi batas luar wajah.'
            )
        },
        {
            'id': 'mouth',
            'name': 'Area Mulut, Gigi & Bibir',
            'icon': 'smile',
            'score': mouth_score,
            'is_edited': mouth_score > 60,
            'status': 'Aktivasi Tinggi' if mouth_score > 70 else ('Aktivasi Sedang' if mouth_score > 50 else 'Aktivasi Normal'),
            'severity': 'Tinggi' if mouth_score > 75 else ('Sedang' if mouth_score > 50 else 'Rendah'),
            'explanation': (
                f'Grad-CAM memetakan intensitas {mouth_score}% pada bibir dan rongga mulut. Filter neural network menangkap variasi gradien tekstur artikulasi bibir.'
            )
        },
        {
            'id': 'skin',
            'name': 'Tekstur Kulit Pipi & Dahi',
            'icon': 'sparkles',
            'score': skin_score,
            'is_edited': skin_score > 60,
            'status': 'Aktivasi Tinggi' if skin_score > 70 else ('Aktivasi Sedang' if skin_score > 50 else 'Aktivasi Normal'),
            'severity': 'Tinggi' if skin_score > 75 else ('Sedang' if skin_score > 50 else 'Rendah'),
            'explanation': (
                f'Intensitas perhatian {skin_score}% terdistribusi pada permukaan kulit pipi dan dahi, mencerminkan ekstraksi fitur frekuensi tekstur pori-pori.'
            )
        }
    ]

    # Menentukan Organ dengan Fokus Tertinggi
    highest_zone_id = max(scores, key=scores.get)
    zone_names = {
        'eyes': 'Area Mata & Alis',
        'jawline': 'Garis Rahang & Batas Wajah',
        'mouth': 'Area Mulut, Gigi & Bibir',
        'skin': 'Tekstur Kulit Pipi & Dahi'
    }
    primary_focus = zone_names.get(highest_zone_id, 'Area Fitur Wajah')
    primary_score = scores[highest_zone_id]

    # Narasi Penjelasan Transparansi Grad-CAM
    xai_summary = (
        f"Peta Grad-CAM menunjukkan bahwa model konvolusi paling memfokuskan perhatiannya pada **{primary_focus}** "
        f"dengan intensitas aktivasi mencapai **{primary_score}%**. Titik fokus tertinggi terpusat pada koordinat "
        f"**X={peak_coord['x']}, Y={peak_coord['y']}** ({peak_name})."
    )

    # 9. Simpan Berkas Artefak
    orig_face_filename = f"{base_output_name}_face.jpg"
    heatmap_filename = f"{base_output_name}_heatmap.jpg"
    overlay_filename = f"{base_output_name}_overlay.jpg"
    
    cv2.imwrite(os.path.join(Config.UPLOAD_FOLDER, orig_face_filename), face_224_bgr)
    cv2.imwrite(os.path.join(Config.UPLOAD_FOLDER, heatmap_filename), heatmap_colored)
    cv2.imwrite(os.path.join(Config.UPLOAD_FOLDER, overlay_filename), overlay_bgr)
    
    elapsed_time = round(time.time() - start_time, 3)
    
    result_package = {
        'verdict': primary_focus,
        'confidence': primary_score,
        'fake_score': primary_score,
        'real_score': round(100.0 - primary_score, 1),
        'primary_focus': primary_focus,
        'primary_score': primary_score,
        'model_name': 'ResNet18-GradCAM-Visualizer',
        'target_layer': 'ResNet.layer4[-1].conv2 (512 Channels)',
        'saved_filename': orig_face_filename,
        'heatmap_filename': heatmap_filename,
        'overlay_filename': overlay_filename,
        'peak_x': peak_coord['x'],
        'peak_y': peak_coord['y'],
        'peak_name': peak_name,
        'peak_intensity': peak_intensity,
        'regions': regions,
        'target_boxes': target_boxes,
        'landmarks': landmarks_224,
        'xai_summary': xai_summary,
        'has_face': has_face,
        'face_box': face_box,
        'technical_details': {
            'execution_time_sec': elapsed_time,
            'input_resolution': '224x224 RGB',
            'gradient_energy': round(grad_energy, 4),
            'fft_energy': round(cues['fft_energy'], 2),
            'laplacian_var': round(cues['laplacian_var'], 2),
            'noise_std': round(cues['noise_std'], 2),
            'colormap': 'OpenCV COLORMAP_JET'
        }
    }
    
    return result_package
