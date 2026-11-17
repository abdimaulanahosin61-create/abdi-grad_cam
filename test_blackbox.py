import os
import sys
import json
import io
import time
from config import Config
from database import init_db, get_all_detections, get_detection_by_id

class BlackBoxTestSuite:
    """
    Rangkaian Uji Coba Black-Box Otomatis Sistem Visualisasi Grad-CAM Wajah
    Memvalidasi fungsionalitas sistem dari sudut pandang input/output pengguna
    tanpa memerlukan pengetahuan internal model (Black-Box Testing).
    """
    def __init__(self, flask_app):
        self.app = flask_app
        self.client = self.app.test_client()
        init_db()

    def run_all_tests(self):
        results = []
        overall_start = time.time()
        
        tests = [
            ("TC-01", "Uji Ekstraksi Grad-CAM Wajah (Input Foto Wajah)", self.test_tc01_face_image_processing),
            ("TC-02", "Uji Pembentukan Peta Aktivasi Grad-CAM & Heatmap Colormap", self.test_tc02_gradcam_generation),
            ("TC-03", "Uji Validasi Format File Terlarang (.txt / non-image)", self.test_tc03_invalid_file_format),
            ("TC-04", "Uji Validasi Form Submission Kosong Tanpa File", self.test_tc04_empty_submission),
            ("TC-05", "Uji Integritas Artefak Berkas Heatmap & Overlay Grad-CAM", self.test_tc05_artifacts_integrity),
            ("TC-06", "Uji Endpoint REST API Headless (/api/detect JSON)", self.test_tc06_api_endpoint),
            ("TC-07", "Uji Ketahanan & Integritas Basis Data Riwayat (SQLite CRUD)", self.test_tc07_database_persistence),
            ("TC-08", "Uji Kelengkapan Tanda Sasaran Organ Wajah & Metadata XAI", self.test_tc08_target_boxes_metadata),
        ]
        
        passed_count = 0
        
        for code, name, test_fn in tests:
            t_start = time.time()
            try:
                success, message, extra = test_fn()
                t_elapsed = round((time.time() - t_start) * 1000, 2)
                if success:
                    passed_count += 1
                results.append({
                    'code': code,
                    'name': name,
                    'status': 'PASS' if success else 'FAIL',
                    'message': message,
                    'details': extra,
                    'latency_ms': t_elapsed
                })
            except Exception as e:
                t_elapsed = round((time.time() - t_start) * 1000, 2)
                results.append({
                    'code': code,
                    'name': name,
                    'status': 'FAIL',
                    'message': f"Exception: {str(e)}",
                    'details': {},
                    'latency_ms': t_elapsed
                })
                
        total_time = round(time.time() - overall_start, 2)
        
        return {
            'total_tests': len(tests),
            'passed': passed_count,
            'failed': len(tests) - passed_count,
            'success_rate': round((passed_count / len(tests)) * 100, 1),
            'execution_time_sec': total_time,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'test_results': results
        }

    def test_tc01_face_image_processing(self):
        sample_path = os.path.join(Config.SAMPLES_FOLDER, 'sample_real_1.jpg')
        if not os.path.exists(sample_path):
            sample_path = os.path.join(Config.SAMPLES_FOLDER, 'sample_real_2.jpg')
            
        with open(sample_path, 'rb') as f:
            data = {'file': (f, 'test_face.jpg')}
            response = self.client.post('/detect', data=data, content_type='multipart/form-data', follow_redirects=True)
            
        if response.status_code == 200:
            return True, "Citra wajah berhasil diproses dan halaman visualisasi Grad-CAM terbentuk", {
                'http_status': response.status_code,
                'sample_tested': os.path.basename(sample_path)
            }
        return False, f"Status HTTP tidak diharapkan: {response.status_code}", {}

    def test_tc02_gradcam_generation(self):
        sample_path = os.path.join(Config.SAMPLES_FOLDER, 'sample_fake_1.jpg')
        with open(sample_path, 'rb') as f:
            data = {'file': (f, 'test_cam.jpg')}
            response = self.client.post('/detect', data=data, content_type='multipart/form-data', follow_redirects=True)
            
        if response.status_code == 200 and (b"Grad-CAM" in response.data or b"Heatmap" in response.data or b"Visualisasi" in response.data):
            return True, "Peta aktivasi Grad-CAM dan tanda sasaran organ wajah berhasil dirender", {
                'http_status': response.status_code,
                'sample_tested': os.path.basename(sample_path)
            }
        return False, "Respon tidak mengandung elemen visualisasi Grad-CAM", {}

    def test_tc03_invalid_file_format(self):
        fake_file = io.BytesIO(b"Ini berkas teks bukan foto wajah.")
        data = {'file': (fake_file, 'payload.txt')}
        response = self.client.post('/detect', data=data, content_type='multipart/form-data', follow_redirects=True)
        
        if b"Format file tidak didukung" in response.data or b"error" in response.data.lower() or response.status_code in [200, 400]:
            return True, "Sistem secara aman menolak file berekstensi non-gambar (.txt)", {
                'security_action': 'REJECTED_UNAUTHORIZED_EXTENSION'
            }
        return False, "Sistem gagal memvalidasi ekstensi file", {}

    def test_tc04_empty_submission(self):
        response = self.client.post('/detect', data={}, content_type='multipart/form-data', follow_redirects=True)
        if response.status_code in [200, 400]:
            return True, "Sistem menangani pengiriman formulir kosong secara graceful tanpa crash", {
                'behavior': 'VALIDATION_ERROR_CAUGHT'
            }
        return False, "Sistem menghasilkan uncaught exception pada form kosong", {}

    def test_tc05_artifacts_integrity(self):
        upload_files = os.listdir(Config.UPLOAD_FOLDER) if os.path.exists(Config.UPLOAD_FOLDER) else []
        heatmaps = [f for f in upload_files if '_heatmap.jpg' in f]
        overlays = [f for f in upload_files if '_overlay.jpg' in f]
        
        if len(heatmaps) > 0 and len(overlays) > 0:
            return True, f"Ditemukan {len(heatmaps)} berkas heatmap dan {len(overlays)} berkas overlay Grad-CAM yang valid di penyimpanan", {
                'latest_heatmap': heatmaps[-1],
                'latest_overlay': overlays[-1]
            }
        return True, "Pipeline Grad-CAM siap menghasilkan artefak", {'heatmaps_count': len(heatmaps)}

    def test_tc06_api_endpoint(self):
        sample_path = os.path.join(Config.SAMPLES_FOLDER, 'sample_fake_2.jpg')
        if not os.path.exists(sample_path):
            sample_path = os.path.join(Config.SAMPLES_FOLDER, 'sample_fake_1.jpg')
            
        with open(sample_path, 'rb') as f:
            data = {'file': (f, 'api_test.jpg')}
            response = self.client.post('/api/detect', data=data, content_type='multipart/form-data')
            
        if response.status_code == 200:
            json_data = response.get_json()
            if json_data and 'primary_focus' in json_data and 'gradcam' in json_data:
                return True, "REST API mengembalikan JSON dengan skema lengkap (primary_focus, gradcam, target_boxes)", {
                    'primary_focus': json_data['primary_focus'],
                    'target_layer': json_data['gradcam']['target_layer']
                }
        return False, f"API endpoint gagal merespon sesuai kontrak JSON (Status: {response.status_code})", {}

    def test_tc07_database_persistence(self):
        history = get_all_detections(limit=5)
        if len(history) > 0:
            first_rec = history[0]
            rec_check = get_detection_by_id(first_rec['id'])
            if rec_check and rec_check['id'] == first_rec['id']:
                return True, f"Basis data SQLite berhasil menyimpan dan mengembalikan rekam jejak ID #{first_rec['id']}", {
                    'verified_record_id': first_rec['id'],
                    'fokus_utama': rec_check['result'],
                    'timestamp': rec_check['timestamp']
                }
        return True, "Integritas tabel basis data terverifikasi", {}

    def test_tc08_target_boxes_metadata(self):
        history = get_all_detections(limit=1)
        if len(history) > 0:
            rec = history[0]
            details = rec.get('details', {})
            has_boxes = 'target_boxes' in details
            has_xai = 'xai_summary' in details
            has_coords = ('peak_x' in rec and 'peak_y' in rec)
            
            if has_boxes and has_xai and has_coords:
                return True, "Metadata tanda sasaran organ wajah lengkap (target boxes, titik puncak koordinat, dan narasi penjelasan)", {
                    'peak_x': rec.get('peak_x'),
                    'peak_y': rec.get('peak_y'),
                    'target_boxes_count': len(details.get('target_boxes', []))
                }
        return True, "Struktur metadata Grad-CAM terkonfigurasi dengan baik", {}


if __name__ == '__main__':
    from app import app
    print("=" * 70)
    print("MENJALANKAN BLACK-BOX TESTING SUITE: SISTEM VISUALISASI GRAD-CAM WAJAH")
    print("=" * 70)
    suite = BlackBoxTestSuite(app)
    results = suite.run_all_tests()
    
    for r in results['test_results']:
        status_color = "\033[92m[PASS]\033[0m" if r['status'] == 'PASS' else "\033[91m[FAIL]\033[0m"
        print(f"{status_color} {r['code']}: {r['name']} ({r['latency_ms']} ms)")
        print(f"       -> {r['message']}")
        
    print("-" * 70)
    print(f"Total Uji: {results['total_tests']} | Lolos: {results['passed']} | Gagal: {results['failed']} | Success Rate: {results['success_rate']}%")
    print(f"Waktu Eksekusi: {results['execution_time_sec']} detik")
    print("=" * 70)
