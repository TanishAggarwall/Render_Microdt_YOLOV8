from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
import cv2
import os
import tempfile
from werkzeug.utils import secure_filename
import base64

app = Flask(__name__, 
           static_folder='../client/build',
           static_url_path='')

# Load YOLOv8 model
model = YOLO('C:/Users/tanis/Desktop/microsofthackathon/models/best (1).pt')

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not supported'}), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        file.save(temp_path)
        
        file_type = filename.rsplit('.', 1)[1].lower()
        
        if file_type in ['png', 'jpg', 'jpeg', 'gif']:
            # Process image
            results = model(temp_path)
            detections = process_image_results(results, temp_path)
            
        elif file_type in ['mp4', 'avi', 'mov', 'mkv']:
            # Process video
            detections = process_video_results(temp_path)
        
        # Clean up temporary file
        os.remove(temp_path)
        
        return jsonify({
            'success': True,
            'detections': detections,
            'file_type': file_type
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_image_results(results, image_path):
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                detection = {
                    'class': result.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].tolist()
                }
                detections.append(detection)
    return detections

def process_video_results(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_results = []
    frame_count = 0
    
    while cap.read()[0] and frame_count < 30:  # Process first 30 frames
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model(frame)
        frame_detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    detection = {
                        'class': result.names[int(box.cls)],
                        'confidence': float(box.conf),
                        'bbox': box.xyxy[0].tolist(),
                        'frame': frame_count
                    }
                    frame_detections.append(detection)
        
        if frame_detections:
            frame_results.extend(frame_detections)
        
        frame_count += 1
    
    cap.release()
    return frame_results

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
