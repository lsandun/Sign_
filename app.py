import os
import csv
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import pytesseract
import re
import pandas as pd
from ultralytics import YOLO

import tensorflow as tf
K = tf.keras.backend

# Set environment variables before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage

from signature_verification import verify_signature, preprocess_image
from signature_extraction import extract_signatures_from_sheet

# --- CONFIG ---
DATASET_FOLDER = 'dataset'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True)

# Load YOLO model
yolo_model = YOLO('models/best.pt')

@tf.keras.utils.register_keras_serializable(name='absolute_difference')
def absolute_difference(inputs):
    x, y = inputs
    return K.abs(x - y)

@tf.keras.utils.register_keras_serializable(name='confidence_penalty_loss')
def confidence_penalty_loss(y_true, y_pred):
    binary_crossentropy = K.binary_crossentropy(y_true, y_pred)
    confidence_penalty = 0.1 * K.binary_crossentropy(K.ones_like(y_pred) * 0.5, y_pred)
    return binary_crossentropy + confidence_penalty

verification_model = tf.keras.models.load_model(
    'models/signature_verification_model.keras',
    custom_objects={
        'absolute_difference': absolute_difference,
        'confidence_penalty_loss': confidence_penalty_loss
    }
)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/download_csv')
def download_csv():
    return send_file(
        os.path.join('static', 'results.csv'),
        mimetype='text/csv',
        as_attachment=True,
        download_name='signature_verification_results.csv'
    )

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # 1. Detect and crop
            regcodes, signatures = detect_and_extract(file_path, app.config['UPLOAD_FOLDER'])
            print(f"YOLO found regcodes: {regcodes}")
            print(f"YOLO found signatures: {signatures}")

            # 2. Associate each signature with the nearest regcode
            associations = associate_signatures_with_regcodes(signatures, regcodes)
            print(f"Associations: {associations}")

            results = []
            for assoc in associations:
                sig_path = assoc['signature_path']
                regcode = assoc['regcode_text']
                print(f"Extracted regcode: {regcode}")
                if not regcode:
                    print(f"Could not extract regcode for signature {sig_path}")
                    continue
                regcode_folder = regcode.replace('/', '_')
                student_folder = os.path.join(DATASET_FOLDER, regcode_folder)
                print(f"Looking for student folder: {student_folder}")
                if not os.path.exists(student_folder):
                    print(f"Student folder missing: {student_folder}")
                    continue
                is_genuine, confidence = verify_with_student_folder(verification_model, sig_path, student_folder)
                results.append({
                    'student_code': regcode,
                    'is_genuine': is_genuine,
                    'confidence': confidence
                })

            # Export to CSV in static folder for download
            df = pd.DataFrame(results)
            csv_path = os.path.join('static', 'results.csv')
            df.to_csv(csv_path, index=False)
            print(f"Results: {results}")
            return render_template('results.html', results=results, csv_file='results.csv')
    return '''
    <!doctype html>
    <title>Upload Signature Sheet</title>
    <h1>Upload Signature Sheet (Image)</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

def process_signature_sheet(sheet_path):
    # Extract signatures and student codes from the sheet
    extracted_data = extract_signatures_from_sheet(sheet_path)
    
    results = []
    
    # For each extracted signature, verify against reference signatures
    for student_code, signature_img in extracted_data:
        # Save the extracted signature temporarily
        temp_sig_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{student_code}_extracted.png")
        cv2.imwrite(temp_sig_path, signature_img)
        
        # Get the folder name for this student's reference signatures
        folder_name = student_code.replace('/', '_')
        
        # Path to genuine signatures for this student
        genuine_folder = os.path.join('dataset', folder_name, 'genuine')
        
        if not os.path.exists(genuine_folder):
            results.append({
                'student_code': student_code,
                'is_genuine': False,
                'confidence': 0.0,
                'message': 'No reference signatures found'
            })
            continue
        
        # Get the first genuine signature as reference
        genuine_signatures = [os.path.join(genuine_folder, f) for f in os.listdir(genuine_folder) 
                             if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if not genuine_signatures:
            results.append({
                'student_code': student_code,
                'is_genuine': False,
                'confidence': 0.0,
                'message': 'No reference signatures found'
            })
            continue
        
        # Verify the signature
        reference_sig = genuine_signatures[0]
        _, is_genuine, confidence = verify_signature(verification_model, reference_sig, temp_sig_path)
        
        results.append({
            'student_code': student_code,
            'is_genuine': is_genuine,
            'confidence': confidence
        })
    
    return results

def detect_and_extract(image_path, output_dir):
    img = cv2.imread(image_path)
    results = yolo_model.predict(img, conf=0.5)
    regcodes = []
    signatures = []
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        for i, (box, cls) in enumerate(zip(boxes, classes)):
            x1, y1, x2, y2 = map(int, box)
            crop = img[y1:y2, x1:x2]
            crop_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_{int(cls)}_{i}.png")
            cv2.imwrite(crop_path, crop)
            if cls == 0:
                # OCR for regcode
                config = '--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/'
                text = pytesseract.image_to_string(crop, config=config).strip()
                # Regex validation
                match = re.match(r'^SEU/IS/20/[A-Z]{2}/\d{3}$', text)
                if match:
                    regcodes.append({'text': text, 'bbox': (x1, y1, x2, y2), 'path': crop_path})
            elif cls == 1:
                signatures.append({'bbox': (x1, y1, x2, y2), 'path': crop_path})
    return regcodes, signatures

def extract_registration_code_from_image(image_path):
    img = cv2.imread(image_path)
    text = pytesseract.image_to_string(img)
    # Use regex to extract registration code pattern
    match = re.search(r'SEU[\/_][A-Z]{2}[\/_]\d{2}[\/_][A-Z]{2}[\/_]\d{3}', text.replace(' ', '').replace('-', ''))
    if match:
        return match.group(0).replace('/', '_').replace('-', '_')
    return None

def associate_signatures_with_regcodes(signatures, regcodes):
    associations = []
    for sig in signatures:
        sig_y = sig['bbox'][1]
        closest_regcode = None
        min_dist = float('inf')
        for reg in regcodes:
            reg_y = reg['bbox'][1]
            dist = abs(sig_y - reg_y)
            if dist < min_dist:
                min_dist = dist
                closest_regcode = reg
        associations.append({'signature_path': sig['path'], 'regcode_text': closest_regcode['text'] if closest_regcode else None})
    return associations

def verify_with_student_folder(model, test_sig_path, student_folder, threshold=0.3):
    genuine_folder = os.path.join(student_folder, 'genuine')
    best_conf = 0
    is_genuine = False
    if not os.path.exists(genuine_folder):
        return False, 0
    for ref_img in os.listdir(genuine_folder):
        ref_path = os.path.join(genuine_folder, ref_img)
        _, is_g, conf = verify_signature(model, ref_path, test_sig_path, threshold)
        if conf > best_conf:
            best_conf = conf
            is_genuine = is_g
    return is_genuine, best_conf

if __name__ == '__main__':
    app.run(debug=True)