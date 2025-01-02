import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
from flask import Flask, render_template, request, jsonify
from predict import AudioPredictor
from datetime import datetime
from werkzeug.utils import secure_filename
import traceback
from pydub import AudioSegment
import numpy as np
import librosa
import requests
import os
import tensorflow as tf

# Force CPU-only operation
tf.config.set_visible_devices([], 'GPU')

app = Flask(__name__)
predictor = AudioPredictor()

# Configure paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'results')
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'webm', 'm4a'}
SEGMENT_LENGTH = 5000  # 5 seconds in milliseconds

# Create necessary directories
for folder in [UPLOAD_FOLDER, RESULTS_FOLDER]:
    os.makedirs(folder, exist_ok=True)
    print(f"Directory created/verified at: {folder}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_segments_folder(audio_filename):
    """Create folders for organizing segments"""
    base_name = os.path.splitext(audio_filename)[0]
    base_folder = os.path.join(RESULTS_FOLDER, base_name)
    fluent_folder = os.path.join(base_folder, 'fluent')
    disfluent_folder = os.path.join(base_folder, 'disfluent')
    
    for folder in [base_folder, fluent_folder, disfluent_folder]:
        os.makedirs(folder, exist_ok=True)
    
    return base_folder, fluent_folder, disfluent_folder

def split_audio(audio_path, segment_length=SEGMENT_LENGTH):
    """Split audio file into segments"""
    try:
        # Load audio file
        audio = AudioSegment.from_file(audio_path)
        
        # Get duration in milliseconds
        duration = len(audio)
        segments = []
        
        # Split audio into segments
        for start in range(0, duration, segment_length):
            end = min(start + segment_length, duration)
            segment = audio[start:end]
            if len(segment) >= 1000:  # Only keep segments longer than 1 second
                segments.append({
                    'audio': segment,
                    'start_time': start,
                    'end_time': end
                })
        
        return segments
    except Exception as e:
        print(f"Error splitting audio: {e}")
        traceback.print_exc()
        return None

def download_model_from_drive():
    os.makedirs('models', exist_ok=True)
    model_path = 'models/model.keras'
    
    if not os.path.exists(model_path):
        # Replace with your direct download link
        download_url = 'https://drive.google.com/file/d/1rcc01FwYJYWA3J2GWw8_gOkqovkXI7tv/view?usp=drive_link'
        
        response = requests.get(download_url)
        with open(model_path, 'wb') as f:
            f.write(response.content)
    
    return model_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        audio_path = os.path.join(UPLOAD_FOLDER, f'audio_{timestamp}.webm')
        
        # Ensure directory exists before saving
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
        
        try:
            audio_file.save(audio_path)
            print(f"Audio file saved to: {audio_path}")
        except Exception as e:
            print(f"Error saving file: {e}")
            return jsonify({'error': f'Error saving file: {str(e)}'}), 500

        predictions = predictor.predict_audio(audio_path)
        
        # Clean up with error handling
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception as e:
            print(f"Error removing file: {e}")
        
        return jsonify(predictions)

    except Exception as e:
        print(f"Error in predict route: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        print("Received API request")
        
        if 'audio' not in request.files:
            return jsonify({
                'status': 'error',
                'error': 'No audio file provided',
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
            }), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({
                'status': 'error',
                'error': 'No audio file selected',
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
            }), 400

        if not allowed_file(audio_file.filename):
            return jsonify({
                'status': 'error',
                'error': f'Invalid file format. Allowed formats: {ALLOWED_EXTENSIONS}',
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
            }), 400

        # Save original audio file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = secure_filename(audio_file.filename)
        original_audio_path = os.path.join(UPLOAD_FOLDER, f'original_{timestamp}_{filename}')
        
        print(f"Saving original file to: {original_audio_path}")
        audio_file.save(original_audio_path)

        # Create folders for segments
        base_folder, fluent_folder, disfluent_folder = create_segments_folder(filename)
        
        # Split audio into segments
        segments = split_audio(original_audio_path)
        if not segments:
            return jsonify({
                'status': 'error',
                'error': 'Error splitting audio file',
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
            }), 500

        # Process each segment
        results = []
        high_confidence_segments = 0  # Counter for segments with confidence >= 0.8
        
        for i, segment in enumerate(segments):
            # Save segment temporarily
            segment_path = os.path.join(UPLOAD_FOLDER, f'segment_{timestamp}_{i}.wav')
            segment['audio'].export(segment_path, format='wav')
            
            try:
                # Get predictions for segment
                predictions = predictor.predict_audio(segment_path)
                
                # Determine segment class and confidence
                prediction_data = predictions.get('keras_model', {})
                is_fluent = prediction_data.get('class') == 'Fluent'
                confidence = prediction_data.get('confidence', 0)
                
                # Only process segments with confidence >= 0.8
                if confidence >= 0.8:
                    high_confidence_segments += 1
                    
                    # Save segment to appropriate folder
                    target_folder = fluent_folder if is_fluent else disfluent_folder
                    segment_filename = f'segment_{i}_{confidence:.2f}.wav'
                    target_path = os.path.join(target_folder, segment_filename)
                    
                    # Move segment to target folder
                    segment['audio'].export(target_path, format='wav')
                    
                    # Add result to list
                    results.append({
                        'segment_number': i,
                        'start_time': segment['start_time'],
                        'end_time': segment['end_time'],
                        'prediction': predictions,
                        'confidence': confidence,
                        'classification': 'Fluent' if is_fluent else 'Disfluent',
                        'saved_path': target_path
                    })
                else:
                    print(f"Skipping segment {i} due to low confidence: {confidence:.2f}")
                
            except Exception as e:
                print(f"Error processing segment {i}: {e}")
            finally:
                # Clean up temporary segment file
                if os.path.exists(segment_path):
                    os.remove(segment_path)

        # Clean up original file
        if os.path.exists(original_audio_path):
            os.remove(original_audio_path)

        return jsonify({
            'status': 'success',
            'filename': filename,
            'total_segments': len(segments),
            'high_confidence_segments': high_confidence_segments,
            'segments_saved': len(results),
            'confidence_threshold': 0.8,
            'results_folder': base_folder,
            'segments': results,
            'timestamp': timestamp
        })

    except Exception as e:
        print(f"Unexpected error in api_predict: {e}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        }), 500

if __name__ == '__main__':
    print(f"Server starting. Upload directory: {UPLOAD_FOLDER}")
    app.run(host='0.0.0.0', port=80)