import librosa
import soundfile
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib

# DataFlair - Emotions in the RAVDESS dataset
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# DataFlair - Emotions to observe
observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

# Create a dictionary to map emotions to numerical values
emotion_to_label = {emotion: i for i, emotion in enumerate(observed_emotions)}
label_to_emotion = {i: emotion for emotion, i in emotion_to_label.items()}

# Load the data and extract features for each sound file
def load_data(test_size=0.2, sample_rate=None):
    x, y = [], []
    max_length = 0
    for file in glob.glob("C:\\Users\\saini\\Downloads\\speech-emotion-recognition-ravdess-data\\Actor_\\.wav"):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        with soundfile.SoundFile(file) as sound_file:
            X = sound_file.read(dtype="float32")
            if sample_rate is None:
                sample_rate = sound_file.samplerate
            max_length = max(max_length, len(X))
            x.append(X)
            y.append(emotion_to_label[emotion])

    # Pad audio files to the maximum length
    x_padded = []
    for audio_file in x:
        padded_audio = np.pad(audio_file, (0, max_length - len(audio_file)), mode='constant')
        x_padded.append(padded_audio)

    # Print information about the data
    print("Total number of audio files:", len(x))
    print("Sample rate of audio files:", sample_rate)
    print("Maximum length of audio files (in samples):", max_length)

    return train_test_split(np.array(x_padded), y, test_size=test_size, random_state=9)

# Get the sample rate from the first audio file
with soundfile.SoundFile(glob.glob("C:\\Users\\saini\\Downloads\\speech-emotion-recognition-ravdess-data\\Actor_\\.wav")[0]) as sound_file:
    sample_rate = sound_file.samplerate

x_train, x_test, y_train, y_test = load_data(test_size=0.25, sample_rate=sample_rate)

print("Number of training samples:", len(x_train))
print("Number of testing samples:", len(x_test))

# Convert audio data to log-mel spectrograms
def compute_log_mel_spectrograms(audio_data, sample_rate, n_mels=128, n_fft=2048, hop_length=512):
    spectrograms = []
    for audio_file in audio_data:
        spectrogram = librosa.feature.melspectrogram(y=audio_file, sr=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        log_mel_spectrogram = librosa.power_to_db(spectrogram)
        spectrograms.append(log_mel_spectrogram)
    return np.array(spectrograms)

X_train_spectrograms = compute_log_mel_spectrograms(x_train, sample_rate)
X_test_spectrograms = compute_log_mel_spectrograms(x_test, sample_rate)

# Reshape and normalize spectrograms for XGBoost
def normalize_spectrograms(spectrograms):
    spectrograms = spectrograms.reshape(len(spectrograms), -1)
    return (spectrograms - np.min(spectrograms)) / (np.max(spectrograms) - np.min(spectrograms))

X_train_spectrograms = normalize_spectrograms(X_train_spectrograms)
X_test_spectrograms = normalize_spectrograms(X_test_spectrograms)

# Print the shape of the spectrograms
print("Shape of X_train_spectrograms:", X_train_spectrograms.shape)
print("Shape of X_test_spectrograms:", X_test_spectrograms.shape)

# Create the XGBoost classifier
xgb_classifier = XGBClassifier(n_estimators=100, max_depth=3, random_state=42)

# Train the XGBoost classifier with evaluation log
eval_set = [(X_train_spectrograms, y_train), (X_test_spectrograms, y_test)]
eval_result = xgb_classifier.fit(X_train_spectrograms, y_train, eval_metric="mlogloss", eval_set=eval_set, verbose=True)

# Save the trained model
joblib.dump(xgb_classifier, "xgb_classifier_model_2.pkl")
print("Model trained and saved successfully.")

# Evaluate the XGBoost classifier
xgb_predictions = xgb_classifier.predict(X_test_spectrograms)
xgb_accuracy = accuracy_score(y_true=y_test, y_pred=xgb_predictions)
print("XGBoost Classifier Accuracy: {:.2f}%".format(xgb_accuracy * 100))

# Plot accuracy vs. epoch
results = xgb_classifier.evals_result()
epochs = len(results['validation_0']['mlogloss'])
x_axis = range(0, epochs)
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
ax.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
plt.show()

# Display sample predictions
num_samples = 50
sample_indices = np.random.choice(len(x_test), num_samples, replace=False)
sample_predictions = xgb_classifier.predict(X_test_spectrograms[sample_indices])

print("Sample Predictions:")
for i, index in enumerate(sample_indices):
    print(f"Audio File {index}: Predicted emotion = {label_to_emotion[sample_predictions[i]]}, True emotion = {label_to_emotion[y_test[index]]}")
