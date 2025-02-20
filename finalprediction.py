import keras
import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
from transformers import pipeline
import tempfile
import soundfile as sf

class LiveVoiceEmotionRecognition:
    def __init__(self, speaker_model_path):
        """
        Initialize the class with the path to the speaker model and Hugging Face's pretrained emotion model.
        - speaker_model_path: Path to the model predicting the speaker (A1, A2, etc.).
        """
        self.speaker_model = self.load_model(speaker_model_path)

        # Load Hugging Face's emotion classification pipeline
        self.emotion_model = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")
        print("Using Hugging Face's pretrained emotion classification model.")
        
    def load_model(self, model_path):
        """
        Load the trained model.
        """
        model = keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}.")
        return model

    def record_audio(self, duration=5, sampling_rate=22050):
        """
        Record live audio from the microphone.
        """
        print("Recording...")
        audio_data = sd.rec(int(duration * sampling_rate), samplerate=sampling_rate, channels=1)
        sd.wait()  # Wait until the recording is finished
        print("Recording completed.")
        return np.squeeze(audio_data)

    def extract_features(self, audio_data, sampling_rate=22050):
        """
        Extract MFCC features from the recorded audio.
        """
        mfccs = np.mean(librosa.feature.mfcc(y=audio_data, sr=sampling_rate, n_mfcc=40).T, axis=0)
        return mfccs

    def make_speaker_prediction(self, features):
        """
        Predict who is speaking based on extracted features.
        """
        x = np.expand_dims(features, axis=0)  # Add batch dimension
        x = np.expand_dims(x, axis=2)         # Add channel dimension
        prediction = np.argmax(self.speaker_model.predict(x), axis=1)[0]
        return self.map_speaker_prediction(prediction)

    def make_emotion_prediction(self, audio_file):
        """
        Predict the emotion of the speaker based on the recorded audio file using Hugging Face's pretrained emotion model.
        """
        result = self.emotion_model(audio_file)
        print(f"Emotion classification result: {result}")
        
        # Extract the scores and find the highest emotion
        scores = [item['score'] for item in result]
        max_index = np.argmax(scores)
        highest_label = result[max_index]['label']
        highest_score = result[max_index]['score']
        
        return highest_label, highest_score

    @staticmethod
    def map_speaker_prediction(prediction):
        """
        Map the model's output to a specific person.
        """
        person_map = {
            0: "A1",
            1: "A2",
            2: "A3",
            3: "A4"
        }
        return person_map.get(prediction, "Unknown")

# Instantiate the class with the speaker model path
recognition = LiveVoiceEmotionRecognition(
    speaker_model_path=r'C:\Users\OMOLP049\Documents\Mritunjay Gupta\New\model\cnn_np(99).keras'
)

# Record and process live audio
audio = recognition.record_audio(duration=5)

# Save the recorded audio to a temporary .wav file
with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmpfile:
    sf.write(tmpfile.name, audio, 22050)  # Save audio data to file
    audio_file = tmpfile.name

# Get speaker prediction
features = recognition.extract_features(audio)
person = recognition.make_speaker_prediction(features)

# Get emotion prediction
emotion, score = recognition.make_emotion_prediction(audio_file)

# Output the results
print(f"The voice belongs to: {person}")
print(f"The emotion detected is: {emotion} with score: {score}")
