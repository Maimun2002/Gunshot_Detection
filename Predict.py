import numpy as np
import librosa
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from scipy.signal import butter, filtfilt

# Zero-phase filter function
def zero_phase_filter(data, cutoff=2000, fs=22050, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Audio preprocessing function
def preprocess_audio(file_path, sr=22050):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        y_filtered = zero_phase_filter(y, cutoff=2000, fs=sr)
        mfccs = librosa.feature.mfcc(y=y_filtered, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y_filtered, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y_filtered, sr=sr)
        zero_crossing = librosa.feature.zero_crossing_rate(y_filtered)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y_filtered, sr=sr)
        combined_features = np.hstack([
            np.mean(mfccs, axis=1),
            np.mean(chroma, axis=1),
            np.mean(spectral_contrast, axis=1),
            np.mean(zero_crossing, axis=1),
            np.mean(spectral_bandwidth, axis=1)
        ])
        return combined_features
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

# Load dataset
def load_dataset(data_dir):
    features = []
    labels = []

    gunshot_folder = os.path.join(data_dir, 'gunshots')
    other_folder = os.path.join(data_dir, 'others')

    for file in os.listdir(gunshot_folder):
        if file.endswith('.wav'):
            file_path = os.path.join(gunshot_folder, file)
            feature = preprocess_audio(file_path)
            if feature is not None:
                features.append(feature)
                labels.append(1)
    
    for file in os.listdir(other_folder):
        if file.endswith('.wav'):
            file_path = os.path.join(other_folder, file)
            feature = preprocess_audio(file_path)
            if feature is not None:
                features.append(feature)
                labels.append(0)

    return np.array(features), np.array(labels)

# Load the trained model and make predictions for new input
def predict_gunshot(model, file_path):
    feature = preprocess_audio(file_path)
    if feature is not None:
        feature = feature.reshape(1, -1)
        prediction = model.predict(feature)
        return "Gunshot" if prediction[0] == 1 else "Not a Gunshot"
    else:
        return "Error processing the file"

# Main workflow
if __name__ == "__main__":
    data_dir = r"D:\JOURNAL\NEW AUDIO"  # Update this path

    features, labels = load_dataset(data_dir)

    if features.size == 0 or labels.size == 0:
        print("No features or labels found. Check your dataset path or preprocessing.")
    else:
        print(f"Loaded {features.shape[0]} samples with {features.shape[1]} features each.")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        # Handle imbalance using SMOTE
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        # Train LightGBM model
        model = LGBMClassifier(class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred))
        print("Recall:", recall_score(y_test, y_pred))
        print("F1 Score:", f1_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        # Predict for a new sound file
        input_file = input("Enter the path to the sound file for prediction: ")
        result = predict_gunshot(model, input_file)
        print(f"Prediction: {result}")
