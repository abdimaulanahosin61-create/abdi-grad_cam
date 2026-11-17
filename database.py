import os
import sqlite3
import json
from datetime import datetime
from config import Config

def get_db_connection():
    conn = sqlite3.connect(Config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_filename TEXT NOT NULL,
            saved_filename TEXT NOT NULL,
            heatmap_filename TEXT NOT NULL,
            overlay_filename TEXT NOT NULL,
            result TEXT NOT NULL,
            confidence REAL NOT NULL,
            fake_score REAL NOT NULL,
            real_score REAL NOT NULL,
            model_name TEXT NOT NULL,
            target_layer TEXT NOT NULL,
            peak_x INTEGER NOT NULL,
            peak_y INTEGER NOT NULL,
            details_json TEXT NOT NULL,
            timestamp DATETIME DEFAULT (datetime('now', 'localtime'))
        )
    ''')
    conn.commit()
    conn.close()

def save_detection(original_filename, saved_filename, heatmap_filename, overlay_filename,
                   result, confidence, fake_score, real_score, model_name, target_layer,
                   peak_x, peak_y, details_dict):
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    details_str = json.dumps(details_dict)
    
    cursor.execute('''
        INSERT INTO detections (
            original_filename, saved_filename, heatmap_filename, overlay_filename,
            result, confidence, fake_score, real_score, model_name, target_layer,
            peak_x, peak_y, details_json, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        original_filename, saved_filename, heatmap_filename, overlay_filename,
        result, float(confidence), float(fake_score), float(real_score),
        model_name, target_layer, int(peak_x), int(peak_y), details_str, now_str
    ))
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id

def get_detection_by_id(record_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM detections WHERE id = ?', (record_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row is None:
        return None
    
    record = dict(row)
    try:
        record['details'] = json.loads(record.get('details_json', '{}'))
    except Exception:
        record['details'] = {}
    return record

def get_all_detections(filter_type='all', search_query='', limit=100):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = 'SELECT * FROM detections'
    conditions = []
    params = []
    
    if search_query:
        conditions.append('(original_filename LIKE ? OR result LIKE ?)')
        params.extend([f"%{search_query}%", f"%{search_query}%"])
        
    if filter_type and filter_type != 'all':
        filter_keywords = {
            'jawline': ['rahang', 'jaw', 'jawline'],
            'eyes': ['mata', 'eye', 'eyes', 'alis'],
            'mouth': ['mulut', 'mouth', 'bibir', 'gigi'],
            'skin': ['kulit', 'skin', 'pipi', 'dahi']
        }
        kw_list = filter_keywords.get(filter_type.lower(), [filter_type])
        clause_parts = []
        for kw in kw_list:
            clause_parts.append('result LIKE ? OR details_json LIKE ?')
            params.extend([f"%{kw}%", f"%{kw}%"])
        conditions.append(f"({' OR '.join(clause_parts)})")
        
    if conditions:
        query += ' WHERE ' + ' AND '.join(conditions)
        
    query += ' ORDER BY id DESC LIMIT ?'
    params.append(limit)
    
    cursor.execute(query, tuple(params))
    rows = cursor.fetchall()
    conn.close()
    
    records = []
    for r in rows:
        rec = dict(r)
        try:
            rec['details'] = json.loads(rec.get('details_json', '{}'))
        except Exception:
            rec['details'] = {}
        records.append(rec)
        
    return records

def delete_detection(record_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Ambil nama file jika ingin membersihkan file lokal
    cursor.execute('SELECT saved_filename, heatmap_filename, overlay_filename FROM detections WHERE id = ?', (record_id,))
    row = cursor.fetchone()
    if row:
        for f in [row['saved_filename'], row['heatmap_filename'], row['overlay_filename']]:
            if f:
                path = os.path.join(Config.UPLOAD_FOLDER, f)
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass
                        
    cursor.execute('DELETE FROM detections WHERE id = ?', (record_id,))
    conn.commit()
    conn.close()

def clear_all_history():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT saved_filename, heatmap_filename, overlay_filename FROM detections')
    rows = cursor.fetchall()
    for row in rows:
        for f in [row['saved_filename'], row['heatmap_filename'], row['overlay_filename']]:
            if f:
                path = os.path.join(Config.UPLOAD_FOLDER, f)
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass
    cursor.execute('DELETE FROM detections')
    conn.commit()
    conn.close()

def get_system_statistics():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*), AVG(confidence) FROM detections')
    count, avg_conf = cursor.fetchone()
    conn.close()
    
    return {
        'total_scans': count or 0,
        'avg_confidence': round(avg_conf, 1) if avg_conf else 0.0
    }
