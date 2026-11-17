# FaceGradCAM: Visualisasi Grad-CAM Citra Wajah Berbasis Explainable AI (XAI)
> **Sistem Informasi & Transparansi AI: Memvisualisasikan Peta Perhatian Jaringan Konvolusional (Grad-CAM) pada Fitur-Fitur Wajah**

Aplikasi web komprehensif berbasis **Python Flask** yang mengimplementasikan visualisasi **Grad-CAM (Gradient-weighted Class Activation Mapping)** untuk memetakan distribusi perhatian jaringan saraf tiruan konvolusional (*Convolutional Neural Network*) pada fitur-fitur wajah secara transparan, akuntabel, dan presisi.

Aplikasi ini berfokus murni pada **Visualisasi Grad-CAM dan Penandaan Organ Wajah** (*Explainable AI*), membuka kotak hitam (*Black-Box*) model kecerdasan buatan untuk melihat di mana neuron konvolusi paling banyak memusatkan perhatiannya.

Proyek ini dibangun secara ketat mengikuti **5 Tahap Alur Sistem**:
1. **Model AI Kamu**: Arsitektur ResNet-18 Deep Learning dengan pengait (*hook*) pada layer konvolusi terakhir (`layer4[-1].conv2`).
2. **Visualisasi Grad-CAM**: Algoritma Explainable AI (XAI) berbasis gradien untuk memetakan heatmap aktivasi organ wajah.
3. **Pengembangan Website Flask**: Integrasi Frontend interaktif dan Backend komputasi inferensi cerdas.
4. **Antarmuka Pengguna (UI/UX)**: Viewer interaktif dilengkapi slider transparansi heatmap (0% - 100%) dan tanda sasaran 4 organ wajah presisi.
5. **Manajemen Sistem & Pengujian**: Pengarsipan database SQLite riwayat visualisasi dan pengujian otomatis *Black-Box Testing* (TC01 s/d TC08).

---

## 🎯 Fitur Utama Sistem

- **🔍 ResNet-18 Convolutional Backbone**: Mengekstraksi representasi fitur konvolusi spasial resolusi tinggi pada `layer4[-1].conv2` (512 feature channels).
- **🔥 Native PyTorch Grad-CAM Computation**: Menghitung bobot neuron $\alpha_k$ melalui *Global Average Pooling* dari gradien kelas terhadap *feature map* $A^k$, kemudian menerapkan *Rectified Linear Unit* (ReLU).
- **👤 Precision Facial Landmark Targeting (YuNet ONNX)**:
  - Melacak koordinat nyata kedua mata, hidung, mulut, dan garis rahang.
  - Memastikan heatmap Grad-CAM mekar tepat di atas organ wajah nyata, bukan di latar belakang atau pakaian.
- **🎨 Interactive Dual-Layer Heatmap Viewer**:
  - Slider transparansi *real-time* (0% Wajah Polos hingga 100% Heatmap Penuh).
  - Pilihan mode tampilan: *Overlay Campuran*, *Citra Asli*, *Heatmap Grad-CAM Murni*, dan *Side-by-Side (Berdampingan)*.
  - Penanda titik puncak koordinat aktivasi $(X_{\text{peak}}, Y_{\text{peak}})$ dengan pin laser.
  - Kotak sasaran presisi (*Target Bounding Boxes*) beranimasi yang membingkai mata, mulut, rahang, dan kulit.
- **👁️ 4 Distribusi Organ Wajah**:
  - *Area Mata & Alis* (Fokus representasi optik kornea dan kelopak mata).
  - *Garis Rahang & Batas Wajah* (Sensitivitas filter konvolusi terhadap kontur tepi wajah).
  - *Area Mulut, Gigi & Bibir* (Gradien tekstur artikulasi bibir).
  - *Tekstur Kulit Pipi & Dahi* (Ekstraksi fitur frekuensi tekstur pori-pori).
- **🧪 Automated Black-Box Testing Suite**:
  - Panel pengujian interaktif langsung di web (`/testing`) dan via CLI terminal (`python test_blackbox.py`).
  - Menjalankan 8 test cases dengan tingkat keberhasilan 100%.
- **📄 Cetak Laporan PDF**: Mode cetak ramah dokumen untuk dokumentasi presentasi dan riset.
- **🌐 RESTful API Headless (`/api/detect`)**: Memungkinkan integrasi pihak ketiga dengan output JSON lengkap.

---

## 📁 Struktur Direktori Proyek

```
Web Gcam/
├── app.py                      # Controller utama Flask, routing, dan REST API
├── config.py                   # Konfigurasi aplikasi, path direktori, dan batas file
├── database.py                 # Manajemen basis data SQLite (CRUD & statistik)
├── gradcam.py                  # Engine komputasi PyTorch Grad-CAM & perender colormap
├── model.py                    # Pipeline AI: deteksi landmark YuNet, ResNet Grad-CAM
├── test_blackbox.py            # Suite pengujian Black-Box otomatis (CLI & Web runner)
├── face_detection_yunet_2023mar.onnx # Model pendeteksi landmark wajah presisi tinggi
├── haarcascade_frontalface_default.xml # Cascade fallback
├── requirements.txt            # Daftar pustaka dependensi Python
├── README.md                   # Dokumentasi teknis proyek
├── static/
│   ├── css/                    # Stylesheet tambahan
│   ├── js/                     # Skrip Javascript interaktif
│   ├── img/
│   │   └── diagram-flow.jpg    # Citra referensi diagram alur sistem
│   ├── samples/                # Foto sampel wajah untuk pengujian instan
│   └── uploads/                # Berkas hasil visualisasi & heatmap
└── templates/
    ├── base.html               # Master layout dengan Tailwind CSS & navigasi
    ├── index.html              # Halaman beranda, KPI statistik, dan 5 tahap alur
    ├── detect.html             # Antarmuka unggah foto wajah
    ├── result.html             # Hasil visualisasi dengan interactive Grad-CAM viewer
    ├── history.html            # Riwayat arsip dengan filter organ wajah
    ├── testing.html            # Dasbor pengujian Black-Box otomatis (Tahap 5)
    └── concept.html            # Penjelasan diagram alur sistem & konsep Explainable AI
```

---

## 🛠️ Instalasi & Menjalankan Aplikasi

### 1. Menyiapkan Virtual Environment
```bash
# Aktivasi di Windows PowerShell:
.venv\Scripts\Activate.ps1
```

### 2. Menginstal Dependensi
```bash
pip install -r requirements.txt
```

### 3. Menjalankan Website Flask
```bash
python app.py
```
Aplikasi akan aktif di:
👉 **`http://localhost:5000`**

---

## 📱 Dukungan Mobile (iOS & Android) & Cara Meng-online-kan

Aplikasi ini telah dioptimalkan secara menyeluruh agar dapat diakses dengan lancar pada perangkat ponsel pintar **iOS (iPhone, iPad)** dan **Android (Samsung, Xiaomi, Oppo, Vivo, Google Pixel, dll.)**:

1. **Auto-Orientasi EXIF Kamera HP**: Foto selfie atau potret yang diambil langsung menggunakan kamera iPhone atau Android sering kali membawa metadata orientasi EXIF 90°. Sistem otomatis memutar (*auto-transpose*) citra agar selalu tegak dan tidak miring.
2. **Dukungan Kamera Langsung**: Tersedia tombol cepat **"Ambil Kamera HP"** untuk memotret wajah langsung menggunakan kamera ponsel, serta **"Buka Galeri Foto"**.
3. **Slider Sentuh Ramah Jari**: Slider transparansi Grad-CAM telah dikonfigurasi dengan *touch target* 26px sehingga nyaman digeser dengan jari pada layar sentuh.
4. **Layout Responsif & Safe-Area**: Kompatibel dengan layar Dynamic Island / notch iPhone dan rasio layar ponsel masa kini.

---

### Cara 1: Membuka di HP via Wi-Fi Lokal (Instan)
1. Pastikan laptop dan HP terhubung ke jaringan **Wi-Fi atau Hotspot yang sama**.
2. Jalankan server dengan klik ganda file `run_app.bat` atau via terminal:
   ```bash
   python app.py
   ```
3. Buka browser **Safari** (iPhone) atau **Chrome** (Android) di HP, lalu ketik alamat IP yang muncul di terminal:
   👉 **`http://10.212.205.127:5000`** *(atau sesuaikan dengan IP laptop Anda)*.

---

### Cara 2: Meng-online-kan ke Publik Secara Gratis (Bisa Diakses dari Mana Saja)

#### Opsi A — Cloudflare Tunnel (Rekomendasi Cepat & Gratis)
1. Unduh executable `cloudflared` dari website resmi Cloudflare.
2. Buka terminal dan jalankan:
   ```bash
   cloudflared tunnel --url http://localhost:5000
   ```
3. Cloudflare akan memberikan link publik resmi bertipe HTTPS (contoh: `https://xxxxxx.trycloudflare.com`). Link ini dapat dibuka oleh siapa saja di iOS maupun Android melalui internet tanpa perlu satu Wi-Fi.

#### Opsi B — Deploy ke Cloud (Render / Railway / Hugging Face)
Repositori ini telah dilengkapi dengan berkas konfigurasi produksi:
- `Procfile` (`web: gunicorn app:app --workers 2 --timeout 120`)
- `requirements.txt` (termasuk `gunicorn`)

Langkah deploy di **Render.com** (Gratis):
1. Unggah kode ke repositori GitHub.
2. Buat akun di [Render.com](https://render.com) dan pilih **New Web Service**.
3. Hubungkan repositori GitHub Anda.
4. Pilih Runtime **Python 3**, Build Command: `pip install -r requirements.txt`, Start Command: `gunicorn app:app`.
5. Website akan online 24/7 dengan domain gratis `https://nama-proyek.onrender.com`.

---

## 🧪 Menjalankan Black-Box Testing (Tahap 5)

Jalankan perintah berikut pada terminal:
```bash
python test_blackbox.py
```
Hasil eksekusi:
```text
======================================================================
MENJALANKAN BLACK-BOX TESTING SUITE: SISTEM VISUALISASI GRAD-CAM WAJAH
======================================================================
[PASS] TC-01: Uji Ekstraksi Grad-CAM Wajah (Input Foto Wajah) (548 ms)
[PASS] TC-02: Uji Pembentukan Peta Aktivasi Grad-CAM & Heatmap (809 ms)
[PASS] TC-03: Uji Validasi Format File Terlarang (.txt) (11 ms)
[PASS] TC-04: Uji Validasi Form Submission Kosong (3 ms)
[PASS] TC-05: Uji Integritas Artefak Berkas Heatmap & Overlay (3 ms)
[PASS] TC-06: Uji Endpoint REST API Headless (/api/detect JSON) (228 ms)
[PASS] TC-07: Uji Ketahanan & Integritas Basis Data SQLite (2 ms)
[PASS] TC-08: Uji Kelengkapan Tanda Sasaran Organ Wajah & XAI (0.8 ms)
----------------------------------------------------------------------
Total Uji: 8 | Lolos: 8 | Gagal: 0 | Success Rate: 100.0%
======================================================================
```
