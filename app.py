import os
import time
import json
import shutil
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from config import Config
from database import (
    init_db,
    save_detection,
    get_detection_by_id,
    get_all_detections,
    delete_detection,
    clear_all_history,
    get_system_statistics
)
from model import run_deepfake_gradcam_pipeline

app = Flask(__name__)
app.config.from_object(Config)

# Pastikan folder penyimpanan tersedia
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.SAMPLES_FOLDER, exist_ok=True)
os.makedirs(Config.IMG_FOLDER, exist_ok=True)

# Inisialisasi basis data
init_db()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

@app.context_processor
def inject_global_data():
    return {
        'project_title': Config.PROJECT_TITLE,
        'system_name': Config.SYSTEM_NAME,
        'current_year': datetime.now().year
    }

# ==============================================================================
# ROUTE 1: BERANDA / HOME
# ==============================================================================
@app.route('/')
def index():
    stats = get_system_statistics()
    recent_scans = get_all_detections(limit=6)
    
    samples = []
    if os.path.exists(Config.SAMPLES_FOLDER):
        sample_files = sorted(os.listdir(Config.SAMPLES_FOLDER))
        for idx, f in enumerate(sample_files):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                samples.append({
                    'filename': f,
                    'label': f"Sampel Citra Wajah #{idx + 1}",
                    'desc': "Visualisasi Aktivasi Organ Wajah"
                })
    return render_template('index.html', stats=stats, recent_scans=recent_scans, samples=samples)

# ==============================================================================
# ROUTE 2: UNGGAH CITRA & VISUALISASI GRAD-CAM
# ==============================================================================
@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Tidak ada file yang dipilih dalam formulir.', 'error')
            return redirect(request.url)
            
        file = request.files['file']
        if file.filename == '':
            flash('Silakan pilih berkas foto wajah terlebih dahulu.', 'error')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            orig_filename = secure_filename(file.filename)
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_output_name = f"{timestamp_str}_{os.path.splitext(orig_filename)[0]}"
            temp_input_path = os.path.join(Config.UPLOAD_FOLDER, f"{base_output_name}_input.jpg")
            file.save(temp_input_path)
            
            try:
                # Eksekusi Pipeline Grad-CAM
                res = run_deepfake_gradcam_pipeline(temp_input_path, base_output_name)
                
                # Simpan Hasil ke Database SQLite
                record_id = save_detection(
                    original_filename=orig_filename,
                    saved_filename=res['saved_filename'],
                    heatmap_filename=res['heatmap_filename'],
                    overlay_filename=res['overlay_filename'],
                    result=res['primary_focus'],
                    confidence=res['confidence'],
                    fake_score=res['fake_score'],
                    real_score=res['real_score'],
                    model_name=res['model_name'],
                    target_layer=res['target_layer'],
                    peak_x=res['peak_x'],
                    peak_y=res['peak_y'],
                    details_dict=res
                )
                
                # Hapus file input temporer
                if os.path.exists(temp_input_path) and temp_input_path != os.path.join(Config.UPLOAD_FOLDER, res['saved_filename']):
                    try:
                        os.remove(temp_input_path)
                    except Exception:
                        pass
                        
                return redirect(url_for('result', id=record_id))
            except Exception as e:
                flash(f"Terjadi kesalahan saat memproses citra: {str(e)}", 'error')
                return redirect(request.url)
        else:
            flash('Format file tidak didukung. Harap unggah berkas PNG, JPG, JPEG, atau WebP.', 'error')
            return redirect(request.url)

    samples = []
    if os.path.exists(Config.SAMPLES_FOLDER):
        sample_files = sorted(os.listdir(Config.SAMPLES_FOLDER))
        for idx, f in enumerate(sample_files):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                samples.append({
                    'filename': f,
                    'label': f"Sampel Wajah #{idx + 1}",
                    'desc': "Klik untuk Visualisasi Grad-CAM"
                })
    return render_template('detect.html', samples=samples)

# ==============================================================================
# ROUTE 2B: UJI MENGGUNAKAN SAMPEL SATU KLIK
# ==============================================================================
@app.route('/detect/sample/<sample_name>', methods=['POST', 'GET'])
def detect_sample(sample_name):
    clean_sample_name = secure_filename(sample_name)
    sample_path = os.path.join(Config.SAMPLES_FOLDER, clean_sample_name)
    
    if not os.path.exists(sample_path):
        flash('Berkas sampel tidak ditemukan.', 'error')
        return redirect(url_for('detect'))
        
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_name = f"{timestamp_str}_{os.path.splitext(clean_sample_name)[0]}"
    temp_input_path = os.path.join(Config.UPLOAD_FOLDER, f"{base_output_name}_input.jpg")
    shutil.copyfile(sample_path, temp_input_path)
    
    try:
        res = run_deepfake_gradcam_pipeline(temp_input_path, base_output_name)
        record_id = save_detection(
            original_filename=clean_sample_name,
            saved_filename=res['saved_filename'],
            heatmap_filename=res['heatmap_filename'],
            overlay_filename=res['overlay_filename'],
            result=res['primary_focus'],
            confidence=res['confidence'],
            fake_score=res['fake_score'],
            real_score=res['real_score'],
            model_name=res['model_name'],
            target_layer=res['target_layer'],
            peak_x=res['peak_x'],
            peak_y=res['peak_y'],
            details_dict=res
        )
        return redirect(url_for('result', id=record_id))
    except Exception as e:
        flash(f"Gagal memproses sampel: {str(e)}", 'error')
        return redirect(url_for('detect'))

# ==============================================================================
# ROUTE 3: HASIL VISUALISASI GRAD-CAM & TANDA SASARAN WAJAH
# ==============================================================================
@app.route('/result/<int:id>')
def result(id):
    record = get_detection_by_id(id)
    if record is None:
        flash('Rekam jejak visualisasi tidak ditemukan atau sudah dihapus.', 'error')
        return redirect(url_for('history'))
        
    details = record.get('details', {})
    regions = details.get('regions', [])
    target_boxes = details.get('target_boxes', [])
    xai_summary = details.get('xai_summary', '')
    technical_details = details.get('technical_details', {})
    
    return render_template(
        'result.html',
        record=record,
        details=details,
        regions=regions,
        target_boxes=target_boxes,
        xai_summary=xai_summary,
        tech=technical_details
    )

# ==============================================================================
# ROUTE 4: RIWAYAT ARSIP VISUALISASI
# ==============================================================================
@app.route('/history')
def history():
    filter_type = request.args.get('filter', 'all')
    search_query = request.args.get('q', '').strip()
    
    records = get_all_detections(filter_type=filter_type, search_query=search_query, limit=100)
    stats = get_system_statistics()
    
    return render_template(
        'history.html',
        records=records,
        stats=stats,
        current_filter=filter_type,
        search_query=search_query
    )

@app.route('/history/delete/<int:id>', methods=['POST'])
def delete_history_item(id):
    delete_detection(id)
    flash(f"Rekam jejak #{id} berhasil dihapus dari sistem.", 'success')
    return redirect(url_for('history'))

@app.route('/history/clear', methods=['POST'])
def clear_history():
    clear_all_history()
    flash("Seluruh riwayat visualisasi Grad-CAM telah berhasil dikosongkan.", 'success')
    return redirect(url_for('history'))

# ==============================================================================
# ROUTE 5: PENGUJIAN BLACK-BOX SISTEM GRAD-CAM
# ==============================================================================
@app.route('/testing')
def testing_dashboard():
    from test_blackbox import BlackBoxTestSuite
    suite = BlackBoxTestSuite(app)
    results = suite.run_all_tests()
    return render_template('testing.html', results=results)

@app.route('/api/run-tests', methods=['POST', 'GET'])
def api_run_tests():
    from test_blackbox import BlackBoxTestSuite
    suite = BlackBoxTestSuite(app)
    results = suite.run_all_tests()
    return jsonify(results)

# ==============================================================================
# ROUTE 6: KONSEP & ALUR SISTEM GRAD-CAM (EXPLAINABLE AI)
# ==============================================================================
@app.route('/concept')
def concept():
    return render_template('concept.html')

# ==============================================================================
# ROUTE 7: REST API HEADLESS ENDPOINT
# ==============================================================================
@app.route('/api/detect', methods=['POST'])
def api_detect():
    """Endpoint REST API untuk pemanggilan visualisasi Grad-CAM headless."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Parameter file tidak ditemukan'}), 400
        
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Format file tidak didukung'}), 400
        
    orig_filename = secure_filename(file.filename)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_name = f"api_{timestamp_str}_{os.path.splitext(orig_filename)[0]}"
    temp_input_path = os.path.join(Config.UPLOAD_FOLDER, f"{base_output_name}_input.jpg")
    file.save(temp_input_path)
    
    try:
        res = run_deepfake_gradcam_pipeline(temp_input_path, base_output_name)
        record_id = save_detection(
            original_filename=orig_filename,
            saved_filename=res['saved_filename'],
            heatmap_filename=res['heatmap_filename'],
            overlay_filename=res['overlay_filename'],
            result=res['primary_focus'],
            confidence=res['confidence'],
            fake_score=res['fake_score'],
            real_score=res['real_score'],
            model_name=res['model_name'],
            target_layer=res['target_layer'],
            peak_x=res['peak_x'],
            peak_y=res['peak_y'],
            details_dict=res
        )
        
        response_payload = {
            'success': True,
            'id': record_id,
            'primary_focus': res['primary_focus'],
            'attention_intensity': res['confidence'],
            'gradcam': {
                'model_name': res['model_name'],
                'target_layer': res['target_layer'],
                'peak_coordinate': {'x': res['peak_x'], 'y': res['peak_y']},
                'peak_name': res['peak_name'],
                'heatmap_url': url_for('static', filename='uploads/' + res['heatmap_filename'], _external=True),
                'overlay_url': url_for('static', filename='uploads/' + res['overlay_filename'], _external=True),
                'cropped_face_url': url_for('static', filename='uploads/' + res['saved_filename'], _external=True)
            },
            'regions': res['regions'],
            'target_boxes': res['target_boxes'],
            'xai_summary': res['xai_summary'],
            'technical_details': res['technical_details']
        }
        return jsonify(response_payload)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stats')
def api_stats():
    return jsonify(get_system_statistics())

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    import socket
    def get_local_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    lan_ip = get_local_ip()
    print("=" * 68)
    print("  🚀 FACE GRAD-CAM WEB SERVER BERJALAN")
    print("=" * 68)
    print(f"  💻 Akses Laptop/PC         : http://localhost:{port}")
    print(f"  📱 Akses HP (iOS / Android): http://{lan_ip}:{port}")
    print("=" * 68)
    print("  Tips: Buka browser Safari (iPhone/iPad) atau Chrome (Android)")
    print(f"  lalu ketikkan alamat: http://{lan_ip}:{port}")
    print("=" * 68)

    # Otomatis membuka browser ke http://localhost:5000 saat dijalankan di terminal
    import time
    import webbrowser
    import threading

    def auto_open_browser():
        time.sleep(1.0)
        webbrowser.open(f"http://localhost:{port}")

    # Buka tepat 1 kali (mencegah tab ganda akibat reloader debug Flask)
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true' or not app.debug:
        threading.Thread(target=auto_open_browser, daemon=True).start()

    app.run(debug=True, host='0.0.0.0', port=port)
