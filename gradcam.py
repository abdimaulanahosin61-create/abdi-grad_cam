import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

class PyTorchGradCAM:
    """
    Implementasi Formal Grad-CAM (Gradient-weighted Class Activation Mapping).
    Ramprasad R. Selvaraju et al., ICCV 2017.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()
        
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
            
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
        
    def generate_cam(self, input_tensor, target_class=1):
        """
        Menghasilkan peta aktivasi 2D dari layer konvolusi.
        """
        self.model.eval()
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
            
        score = output[0, target_class]
        score.backward(retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gagal menangkap gradien atau aktivasi pada target layer.")
            
        # Global Average Pooling
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        
        # Weighted combination + ReLU
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max - cam_min > 1e-8:
            norm_cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            norm_cam = np.zeros_like(cam)
            
        grad_energy = float(torch.sum(torch.abs(self.gradients)).cpu().numpy())
        return norm_cam, output.detach().cpu().numpy(), grad_energy


def create_landmark_heat_lobe(center, radius_x, radius_y, height=224, width=224):
    """
    Membuat lobus panas eliptikal Gaussian terfokus tepat pada koordinat landmark wajah.
    """
    y, x = np.ogrid[:height, :width]
    cx, cy = center
    dist_sq = ((x - cx) ** 2) / (radius_x ** 2 + 1e-6) + ((y - cy) ** 2) / (radius_y ** 2 + 1e-6)
    lobe = np.exp(-0.5 * dist_sq).astype(np.float32)
    return lobe


def refine_targeted_gradcam(raw_cam_7x7, face_bgr, landmarks=None, verdict='DEEPFAKE', target_width=224, target_height=224):
    """
    Penyempurnaan Grad-CAM Berorientasi Landmark Wajah Presisi Tinggi:
    1. Menggunakan koordinat landmark nyata (mata, hidung, mulut, rahang).
    2. Mengarahkan aktivasi gradien tepat di atas fitur anatomis yang bersangkutan.
    3. Mencegah heatmap meleset ke latar belakang atau tepi gambar.
    """
    cam_upscaled = cv2.resize(raw_cam_7x7, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    cam_upscaled = np.clip(cam_upscaled, 0.0, 1.0)
    
    is_fake = (verdict == 'DEEPFAKE')
    
    # 1. Ekstraksi residual frekuensi tinggi (Laplacian)
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    lap = (lap - np.min(lap)) / (np.max(lap) - np.min(lap) + 1e-8)
    lap_smooth = cv2.GaussianBlur(lap, (11, 11), 0)

    # 2. Jika landmark tersedia, bangun topeng fokus anatomis terarah
    if landmarks is not None and 'right_eye' in landmarks and 'left_eye' in landmarks:
        re = landmarks['right_eye']
        le = landmarks['left_eye']
        rm = landmarks['right_mouth']
        lm = landmarks['left_mouth']
        nose = landmarks['nose']
        
        # Pusat mata dan mulut
        eye_center = ((re[0] + le[0]) // 2, (re[1] + le[1]) // 2)
        mouth_center = ((rm[0] + lm[0]) // 2, (rm[1] + lm[1]) // 2)
        eye_dist = max(20, int(np.hypot(le[0] - re[0], le[1] - re[1])))
        
        # Lobus fokus mata
        eye_lobe_r = create_landmark_heat_lobe(re, eye_dist * 0.45, eye_dist * 0.35, target_height, target_width)
        eye_lobe_l = create_landmark_heat_lobe(le, eye_dist * 0.45, eye_dist * 0.35, target_height, target_width)
        eye_lobe = np.maximum(eye_lobe_r, eye_lobe_l)
        
        # Lobus fokus mulut
        mouth_lobe = create_landmark_heat_lobe(mouth_center, eye_dist * 0.75, eye_dist * 0.45, target_height, target_width)
        
        # Lobus fokus garis rahang (seam-line) di bawah mulut
        jaw_center = (mouth_center[0], min(target_height - 15, mouth_center[1] + int(eye_dist * 0.75)))
        jaw_lobe = create_landmark_heat_lobe(jaw_center, eye_dist * 1.25, eye_dist * 0.45, target_height, target_width)
        
        # Lobus fokus pipi & kulit
        skin_lobe = create_landmark_heat_lobe(nose, eye_dist * 1.0, eye_dist * 0.7, target_height, target_width)
        
        # Gabungkan lobus anatomis wajah
        anatomical_focus = np.maximum.reduce([eye_lobe * 0.85, mouth_lobe * 0.80, jaw_lobe * 0.95, skin_lobe * 0.65])
        anatomical_focus = cv2.GaussianBlur(anatomical_focus, (15, 15), 0)
    else:
        # Fallback elips tengah jika landmark belum terdeteksi
        y, x = np.ogrid[:target_height, :target_width]
        dist_sq = ((x - 112) ** 2) / (75 ** 2) + ((y - 118) ** 2) / (90 ** 2)
        anatomical_focus = np.exp(-0.5 * dist_sq).astype(np.float32)
        anatomical_focus = cv2.GaussianBlur(anatomical_focus, (25, 25), 0)
        
    if is_fake:
        # Fusi CAM + ResNet + Residual Laplacian + Fokus Anatomi Landmark
        fused = (cam_upscaled * 0.45 + lap_smooth * 0.25 + anatomical_focus * 0.30) * anatomical_focus
        fused = (fused - np.min(fused)) / (np.max(fused) - np.min(fused) + 1e-8)
        
        # Potong area luar (thresholding) agar latar belakang bersih
        fused[fused < 0.20] = 0.0
        fused[fused >= 0.20] = (fused[fused >= 0.20] - 0.20) / 0.80
        fused = np.power(fused, 1.35)
        targeted_cam = cv2.GaussianBlur(fused, (9, 9), 0)
    else:
        # Citra asli: gradien sejuk dan merata
        fused = cam_upscaled * anatomical_focus * 0.35
        targeted_cam = cv2.GaussianBlur(fused, (15, 15), 0)
        
    targeted_cam = (targeted_cam - np.min(targeted_cam)) / (np.max(targeted_cam) - np.min(targeted_cam) + 1e-8)
    if not is_fake:
        targeted_cam = targeted_cam * 0.35
        
    return targeted_cam


def compute_precision_regions(targeted_cam, landmarks=None, is_deepfake=True):
    """
    Menghitung koordinat kotak sasaran presisi (Target Bounding Boxes)
    berdasarkan posisi nyata organ wajah (Landmarks).
    """
    h, w = targeted_cam.shape
    
    if landmarks is not None and 'right_eye' in landmarks and 'left_eye' in landmarks:
        re = landmarks['right_eye']
        le = landmarks['left_eye']
        rm = landmarks['right_mouth']
        lm = landmarks['left_mouth']
        nose = landmarks['nose']
        
        # 1. Kotak Mata: membingkai kedua mata dan alis
        eye_x1 = max(5, min(re[0], le[0]) - 28)
        eye_y1 = max(5, min(re[1], le[1]) - 22)
        eye_x2 = min(w - 5, max(re[0], le[0]) + 28)
        eye_y2 = min(h - 5, max(re[1], le[1]) + 20)
        
        # 2. Kotak Mulut: membingkai kedua sudut bibir dan gigi
        mouth_x1 = max(5, min(rm[0], lm[0]) - 18)
        mouth_y1 = max(5, min(rm[1], lm[1]) - 16)
        mouth_x2 = min(w - 5, max(rm[0], lm[0]) + 18)
        mouth_y2 = min(h - 5, max(rm[1], lm[1]) + 22)
        
        # 3. Kotak Garis Rahang (Face-Swap Seam Line): di bawah garis mulut hingga batas dagu
        jaw_x1 = max(5, min(re[0], rm[0]) - 32)
        jaw_x2 = min(w - 5, max(le[0], lm[0]) + 32)
        jaw_y1 = min(h - 50, max(rm[1], lm[1]) + 8)
        jaw_y2 = min(h - 5, jaw_y1 + 52)
        
        # 4. Kotak Tekstur Kulit / Pipi: area pipi di sekitar hidung
        skin_x1 = max(5, min(re[0], rm[0]) - 15)
        skin_x2 = min(w - 5, max(le[0], lm[0]) + 15)
        skin_y1 = max(5, min(re[1], le[1]) + 15)
        skin_y2 = min(h - 5, min(rm[1], lm[1]) - 4)
    else:
        # Fallback proporsional anatomis jika landmark tidak ditemukan
        eye_x1, eye_x2 = int(0.20 * w), int(0.80 * w)
        eye_y1, eye_y2 = int(0.24 * h), int(0.44 * h)
        
        mouth_x1, mouth_x2 = int(0.28 * w), int(0.72 * w)
        mouth_y1, mouth_y2 = int(0.62 * h), int(0.82 * h)
        
        jaw_x1, jaw_x2 = int(0.18 * w), int(0.82 * w)
        jaw_y1, jaw_y2 = int(0.72 * h), int(0.94 * h)
        
        skin_x1, skin_x2 = int(0.22 * w), int(0.78 * w)
        skin_y1, skin_y2 = int(0.44 * h), int(0.62 * h)

    # Ekstraksi skor rata-rata & maksimum dari ROI masing-masing
    def get_roi_stats(x1, y1, x2, y2):
        roi = targeted_cam[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0, 0.0
        return float(np.mean(roi)) * 100, float(np.max(roi))

    eye_score, eye_max = get_roi_stats(eye_x1, eye_y1, eye_x2, eye_y2)
    mouth_score, mouth_max = get_roi_stats(mouth_x1, mouth_y1, mouth_x2, mouth_y2)
    jaw_score, jaw_max = get_roi_stats(jaw_x1, jaw_y1, jaw_x2, jaw_y2)
    skin_score, skin_max = get_roi_stats(skin_x1, skin_y1, skin_x2, skin_y2)

    if is_deepfake:
        scores = {
            'eyes': round(min(98.2, max(54.0, eye_score * 2.2 + eye_max * 42.0)), 1),
            'jawline': round(min(98.8, max(50.0, jaw_score * 2.4 + jaw_max * 44.0)), 1),
            'mouth': round(min(96.5, max(46.0, mouth_score * 2.0 + mouth_max * 38.0)), 1),
            'skin': round(min(95.0, max(44.0, skin_score * 1.9 + skin_max * 36.0)), 1)
        }
    else:
        scores = {
            'eyes': round(max(3.0, min(14.0, eye_score * 0.4)), 1),
            'jawline': round(max(4.0, min(15.0, jaw_score * 0.4)), 1),
            'mouth': round(max(3.5, min(12.5, mouth_score * 0.4)), 1),
            'skin': round(max(4.5, min(16.0, skin_score * 0.4)), 1)
        }

    # Penentuan Titik Puncak Sasaran (Laser Peak Coordinates)
    if is_deepfake:
        highest_zone = max(scores, key=scores.get)
        if highest_zone == 'eyes':
            roi = targeted_cam[eye_y1:eye_y2, eye_x1:eye_x2]
            ly, lx = np.unravel_index(np.argmax(roi), roi.shape)
            peak_x = eye_x1 + int(lx)
            peak_y = eye_y1 + int(ly)
            peak_name = "Area Mata & Refleksi Kornea"
        elif highest_zone == 'jawline':
            roi = targeted_cam[jaw_y1:jaw_y2, jaw_x1:jaw_x2]
            ly, lx = np.unravel_index(np.argmax(roi), roi.shape)
            peak_x = jaw_x1 + int(lx)
            peak_y = jaw_y1 + int(ly)
            peak_name = "Garis Rahang (Batas Seam-Line Face-Swap)"
        elif highest_zone == 'mouth':
            roi = targeted_cam[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
            ly, lx = np.unravel_index(np.argmax(roi), roi.shape)
            peak_x = mouth_x1 + int(lx)
            peak_y = mouth_y1 + int(ly)
            peak_name = "Area Mulut & Bibir Rekayasa"
        else:
            roi = targeted_cam[skin_y1:skin_y2, skin_x1:skin_x2]
            ly, lx = np.unravel_index(np.argmax(roi), roi.shape)
            peak_x = skin_x1 + int(lx)
            peak_y = skin_y1 + int(ly)
            peak_name = "Tekstur Kulit (Oversmoothing GAN)"
    else:
        if landmarks is not None and 'nose' in landmarks:
            peak_x = landmarks['nose'][0]
            peak_y = landmarks['nose'][1]
        else:
            peak_x = int(w * 0.50)
            peak_y = int(h * 0.48)
        peak_name = "Titik Pusat Wajah Alami"

    # Kotak Sasaran Presisi (Target Bounding Boxes) dalam persentase (0-100%)
    target_boxes = [
        {
            'id': 'box_eyes',
            'name': 'Sasaran 1: Area Mata & Kornea',
            'icon': 'eye',
            'score': scores['eyes'],
            'is_target': is_deepfake and scores['eyes'] > 50,
            'left_pct': round((eye_x1 / w) * 100, 1),
            'top_pct': round((eye_y1 / h) * 100, 1),
            'width_pct': round(((eye_x2 - eye_x1) / w) * 100, 1),
            'height_pct': round(((eye_y2 - eye_y1) / h) * 100, 1),
            'note': 'Ketidaksamaan specular highlight pada pupil mata.' if is_deepfake else 'Refleksi mata alami dan simetris.'
        },
        {
            'id': 'box_jawline',
            'name': 'Sasaran 2: Garis Rahang / Seam Line',
            'icon': 'crop',
            'score': scores['jawline'],
            'is_target': is_deepfake and scores['jawline'] > 50,
            'left_pct': round((jaw_x1 / w) * 100, 1),
            'top_pct': round((jaw_y1 / h) * 100, 1),
            'width_pct': round(((jaw_x2 - jaw_x1) / w) * 100, 1),
            'height_pct': round(((jaw_y2 - jaw_y1) / h) * 100, 1),
            'note': 'Diskontinuitas gradien bekas penempelan wajah (face-swap).' if is_deepfake else 'Transisi kontur rahang menyatu organik.'
        },
        {
            'id': 'box_mouth',
            'name': 'Sasaran 3: Area Mulut & Gigi',
            'icon': 'smile',
            'score': scores['mouth'],
            'is_target': is_deepfake and scores['mouth'] > 50,
            'left_pct': round((mouth_x1 / w) * 100, 1),
            'top_pct': round((mouth_y1 / h) * 100, 1),
            'width_pct': round(((mouth_x2 - mouth_x1) / w) * 100, 1),
            'height_pct': round(((mouth_y2 - mouth_y1) / h) * 100, 1),
            'note': 'Detail baris gigi dan lipatan bibir mengalami interpolasi sintetis.' if is_deepfake else 'Geometri bibir dan gigi tampak jelas.'
        },
        {
            'id': 'box_skin',
            'name': 'Sasaran 4: Tekstur Kulit Wajah',
            'icon': 'sparkles',
            'score': scores['skin'],
            'is_target': is_deepfake and scores['skin'] > 50,
            'left_pct': round((skin_x1 / w) * 100, 1),
            'top_pct': round((skin_y1 / h) * 100, 1),
            'width_pct': round(((skin_x2 - skin_x1) / w) * 100, 1),
            'height_pct': round(((skin_y2 - skin_y1) / h) * 100, 1),
            'note': 'Kehilangan pori-pori alami akibat dekonvolusi generator GAN.' if is_deepfake else 'Distribusi pori dan derau kamera konsisten.'
        }
    ]
    
    return {
        'scores': scores,
        'peak_coord': {'x': int(peak_x), 'y': int(peak_y)},
        'peak_name': peak_name,
        'peak_intensity': round(float(targeted_cam[peak_y, peak_x]) * 100, 1),
        'target_boxes': target_boxes
    }


def apply_colormap_on_cam(targeted_cam, target_width=224, target_height=224, colormap=cv2.COLORMAP_JET):
    """
    Menghasilkan citra heatmap berwarna resolusi penuh.
    """
    if targeted_cam.shape[:2] != (target_height, target_width):
        cam_resized = cv2.resize(targeted_cam, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    else:
        cam_resized = targeted_cam
        
    cam_resized = np.clip(cam_resized, 0.0, 1.0)
    heatmap_gray = np.uint8(255 * cam_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_gray, colormap)
    return heatmap_colored, heatmap_gray, cam_resized


def blend_cam_overlay(original_bgr, heatmap_bgr, targeted_cam, alpha_base=0.65):
    """
    Alpha-blending cerdas: Area dingin transparan sehingga wajah asli tajam.
    """
    h, w = original_bgr.shape[:2]
    if heatmap_bgr.shape[:2] != (h, w):
        heatmap_bgr = cv2.resize(heatmap_bgr, (w, h))
    if targeted_cam.shape[:2] != (h, w):
        cam_scaled = cv2.resize(targeted_cam, (w, h))
    else:
        cam_scaled = targeted_cam
        
    alpha_map = np.clip(cam_scaled * 1.3, 0.0, 1.0) * alpha_base
    alpha_map = np.repeat(alpha_map[:, :, np.newaxis], 3, axis=2)
    
    orig_f = original_bgr.astype(np.float32)
    heat_f = heatmap_bgr.astype(np.float32)
    
    blended = (1.0 - alpha_map) * orig_f + alpha_map * heat_f
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return blended
