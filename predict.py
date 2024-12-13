import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import librosa
from tensorflow.keras.models import load_model
import soundfile as sf
import os
import tensorflow as tf
import traceback
import json
from datetime import datetime
from pydub import AudioSegment
import io
import requests
import gdown
import gc

# Configure TensorFlow for memory optimization
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class AudioPredictor:
    def _load_keras_model(self, path):
        """Load .keras format model"""
        try:
            if os.path.exists(path):
                print(f"Found Keras model at: {path}")
                model = load_model(path)
                print("Successfully loaded Keras model")
                return model
            else:
                print(f"Keras model not found at: {path}")
            return None
        except Exception as e:
            print(f"Error loading Keras model: {str(e)}")
            traceback.print_exc()
            return None

    def _load_h5_model(self, path):
        """Load .h5 format model"""
        try:
            return load_model(path) if os.path.exists(path) else None
        except Exception as e:
            print(f"Error loading .h5 model: {e}")
            return None

    def _load_saved_model(self, path):
        """Load SavedModel format"""
        try:
            if os.path.exists(path):
                print(f"Found SavedModel directory at: {path}")
                print("Contents of SavedModel directory:")
                for item in os.listdir(path):
                    print(f"- {item}")
                
                # Load the saved model
                model = tf.keras.models.load_model(path)
                print(f"Successfully loaded SavedModel from {path}")
                return model
            else:
                print(f"SavedModel directory not found at: {path}")
            return None
        except Exception as e:
            print(f"Error loading SavedModel: {str(e)}")
            traceback.print_exc()
            return None

    def _load_architecture_weights(self, arch_path, weights_path):
        """Load model from architecture JSON and weights"""
        try:
            if os.path.exists(arch_path) and os.path.exists(weights_path):
                with open(arch_path, 'r') as f:
                    # Load and parse JSON string
                    model_json = f.read()
                    model = tf.keras.models.model_from_json(model_json)
                    model.load_weights(weights_path)
                    return model
            return None
        except Exception as e:
            print(f"Error loading architecture and weights: {e}")
            return None

    def _download_model(self):
        """Download model from Google Drive if not present"""
        try:
            models_dir = 'models'
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, 'model.keras')
            
            if not os.path.exists(model_path):
                print("Model not found locally. Downloading from Google Drive...")
                
                # The file ID from your Google Drive share link
                file_id = '1rcc01FwYJYWA3J2GWw8_gOkqovkXI7tv'
                
                # Construct the download URL
                url = f'https://drive.google.com/uc?export=download&id={file_id}'
                
                try:
                    print(f"Attempting to download from: {url}")
                    gdown.download(url, model_path, quiet=False, fuzzy=True)
                    
                    if not os.path.exists(model_path):
                        raise Exception("Download completed but file not found")
                        
                    print(f"Model downloaded successfully to {model_path}")
                except Exception as download_error:
                    print(f"Download error: {download_error}")
                    raise
            else:
                print(f"Model already exists at {model_path}")
            
            return model_path
        except Exception as e:
            print(f"Error in _download_model: {str(e)}")
            raise

    def __init__(self):
        print("Loading models...")
        try:
            self.model = None
            self.sample_rate = 16000

            # Get the models directory and download if needed
            model_path = self._download_model()
            
            # Load the Keras model with memory optimization
            tf.keras.backend.clear_session()
            self.model = self._load_keras_model(model_path)
            if self.model is None:
                raise ValueError("Failed to load model.keras")
            
            self.models = {'keras_model': self.model}
            print("Successfully loaded Keras model")

            # Force garbage collection
            gc.collect()

            # Try to load test data if available (but don't require it)
            try:
                self.X_test = np.load(os.path.join(models_dir, 'X_test.npy'))
                self.y_test = np.load(os.path.join(models_dir, 'y_test.npy'))
                print("\nTest data loaded successfully")
            except Exception as e:
                print(f"\nNote: Test data not available: {e}")
                self.X_test = None
                self.y_test = None

            print(f"\nSuccessfully loaded model")
            print(f"Model type: Keras model")

        except Exception as e:
            print(f"\nError loading model: {e}")
            traceback.print_exc()
            raise

    def predict_audio(self, audio_path):
        """Predict fluency from audio file using all available models"""
        try:
            # Convert webm to wav
            audio = AudioSegment.from_file(audio_path, format="webm")
            wav_path = audio_path.rsplit('.', 1)[0] + '.wav'
            audio.export(wav_path, format="wav")

            # Load and preprocess audio
            audio, sr = sf.read(wav_path)
            print(f"Loaded audio shape: {audio.shape}, sample rate: {sr}")

            # Clean up temporary wav file
            os.remove(wav_path)

            # Ensure mono audio
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            # Ensure correct sample rate
            if sr != self.sample_rate:
                audio = librosa.resample(
                    audio, orig_sr=sr, target_sr=self.sample_rate)

            # Normalize audio
            audio = audio / np.max(np.abs(audio))

            # Get mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_mels=131,
                fmax=8000
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # Ensure correct shape (131, 130)
            if mel_spec_db.shape[1] != 130:
                if mel_spec_db.shape[1] < 130:
                    pad_width = 130 - mel_spec_db.shape[1]
                    mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)))
                else:
                    mel_spec_db = mel_spec_db[:, :130]

            # Make predictions using all available models
            predictions = {}
            for model_name, model in self.models.items():
                if model is not None:
                    # Clear session before prediction
                    tf.keras.backend.clear_session()
                    
                    # Make prediction
                    pred = model.predict(np.expand_dims(
                        mel_spec_db, axis=0), verbose=0)
                    class_names = ['Fluent', 'Disfluent']
                    predicted_class = class_names[np.argmax(pred)]
                    confidence = float(np.max(pred))

                    prediction_dict = {
                        'class': predicted_class,
                        'confidence': confidence,
                        'raw_output': pred.tolist()
                    }

                    # Add test metrics if test data is available
                    try:
                        if hasattr(self, 'X_test') and hasattr(self, 'y_test') and \
                           self.X_test is not None and self.y_test is not None:
                            test_loss, test_accuracy = model.evaluate(
                                self.X_test, self.y_test, verbose=0)
                            prediction_dict.update({
                                'test_accuracy': float(test_accuracy),
                                'test_loss': float(test_loss)
                            })
                    except Exception as e:
                        print(
                            f"Warning: Could not evaluate {model_name} on test data: {e}")

                    predictions[model_name] = prediction_dict

            # Force garbage collection after predictions
            gc.collect()
            return predictions

        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            traceback.print_exc()
            return {"error": str(e)}

    def verify_model(self):
        """Verify model architecture and weights"""
        try:
            # Create a small test input
            test_input = np.random.random((1, 131, 130))

            # Get prediction
            test_pred = self.model.predict(test_input, verbose=0)

            print("\nModel Verification:")
            print(f"Test input shape: {test_input.shape}")
            print(f"Test output shape: {test_pred.shape}")
            print(f"Test output values: {test_pred}")
            print(
                f"Test prediction: {['Fluent', 'Disfluent'][np.argmax(test_pred)]}")

            # Print model summary
            print("\nModel Architecture:")
            self.model.summary()

            return True
        except Exception as e:
            print(f"Model verification failed: {e}")
            return False


# Example usage
if __name__ == "__main__":
    predictor = AudioPredictor()

    # Verify all loaded models
    print("\nVerifying models...")
    verification_results = predictor.verify_model()
    if verification_results:
        for model_name, results in verification_results.items():
            print(f"\n{model_name} verification results:")
            for key, value in results.items():
                print(f"{key}: {value}")
