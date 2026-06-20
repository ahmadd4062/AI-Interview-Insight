import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from deepface import DeepFace

# ----------------------------
# === FILE PATHS ===
# ----------------------------
audio_confident = r'Dataset\Ravdess_Confident'
audio_nervous = r'Dataset\Ravdess_Nervous'
image_confident = r'Facial Dataset\Confident'
image_nervous = r'Facial Dataset\Nervous'

# ----------------------------
# === AUDIO FEATURE EXTRACTOR ===
# ----------------------------
def extract_audio_features(file_path):
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=audio).T, axis=0)
    return np.hstack([mfccs, chroma, zcr, rms])

# ----------------------------
# === LOAD AUDIO DATA ===
# ----------------------------
audio_features = []
audio_labels = []

for label, directory in [(0, audio_confident), (1, audio_nervous)]:
    for file in os.listdir(directory):
        if file.endswith(".wav"):
            path = os.path.join(directory, file)
            feature = extract_audio_features(path)
            audio_features.append(feature)
            audio_labels.append(label)

X_audio = np.array(audio_features)
y_audio = np.array(audio_labels)

# Train-test split and scaling
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_audio, y_audio, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_a_scaled = scaler.fit_transform(X_train_a)
X_test_a_scaled = scaler.transform(X_test_a)

# ----------------------------
# === AUDIO MODELS ===
# ----------------------------
def evaluate_audio_model(name, y_true, y_pred):
    print(f"\n{name} Accuracy: {accuracy_score(y_true, y_pred)*100:.2f}%")
    print(classification_report(y_true, y_pred, target_names=["Confident", "Nervous"]))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Confident", "Nervous"], yticklabels=["Confident", "Nervous"])
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    return accuracy_score(y_true, y_pred)

# Train and evaluate
models_audio = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(kernel='linear', probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

for name, model in models_audio.items():
    if name in ["SVM", "KNN"]:
        model.fit(X_train_a_scaled, y_train_a)
        y_pred = model.predict(X_test_a_scaled)
    else:
        model.fit(X_train_a, y_train_a)
        y_pred = model.predict(X_test_a)
    evaluate_audio_model(name, y_test_a, y_pred)

# ----------------------------
# === IMAGE PROCESSING ===
# ----------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def map_emotion(emotion):
    return 'Confident' if emotion in ['happy', 'neutral'] else 'Nervous'

def process_image_data():
    deepface_true, deepface_pred = [], []
    haar_true, haar_pred = [], []
    X_faces, y_faces = [], []

    for label_dir, true_label in [(image_confident, "Confident"), (image_nervous, "Nervous")]:
        for file in os.listdir(label_dir):
            if file.endswith(('.jpg', '.png')):
                path = os.path.join(label_dir, file)
                img = cv2.imread(path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # --- DeepFace ---
                try:
                    analysis = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
                    emotion = analysis[0]['dominant_emotion']
                    pred_label = map_emotion(emotion)
                except:
                    pred_label = "Nervous"
                deepface_true.append(true_label)
                deepface_pred.append(pred_label)

                # --- Haarcascade Brightness ---
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    face = gray[y:y+h, x:x+w]
                else:
                    face = cv2.resize(gray, (48, 48))
                avg_brightness = np.mean(face)
                haar_pred_label = "Confident" if avg_brightness > 100 else "Nervous"
                haar_true.append(true_label)
                haar_pred.append(haar_pred_label)

                # --- Prepare for Logistic Regression ---
                face = cv2.resize(face, (48, 48))
                X_faces.append(face.flatten())
                y_faces.append(0 if true_label == 'Confident' else 1)

    return deepface_true, deepface_pred, haar_true, haar_pred, np.array(X_faces), np.array(y_faces)

# Run image processing
deepface_true, deepface_pred, haar_true, haar_pred, X_faces, y_faces = process_image_data()

# ----------------------------
# === IMAGE MODEL EVALUATION ===
# ----------------------------
def show_metrics(y_true, y_pred, model_name):
    print(f"\n--- {model_name} ---")
    print(classification_report(y_true, y_pred, target_names=["Confident", "Nervous"]))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Confident", "Nervous"], yticklabels=["Confident", "Nervous"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    print(f"Accuracy: {accuracy*100:.2f}%")
    return accuracy

# Convert string labels to numeric
deepface_true_num = [0 if lbl == 'Confident' else 1 for lbl in deepface_true]
deepface_pred_num = [0 if lbl == 'Confident' else 1 for lbl in deepface_pred]
haar_true_num = [0 if lbl == 'Confident' else 1 for lbl in haar_true]
haar_pred_num = [0 if lbl == 'Confident' else 1 for lbl in haar_pred]

# DeepFace & Haar
accuracy_deepface = show_metrics(deepface_true_num, deepface_pred_num, "DeepFace")
accuracy_haar = show_metrics(haar_true_num, haar_pred_num, "Haarcascade")

# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_faces, y_faces)
y_pred_log = log_model.predict(X_faces)
accuracy_logistic = show_metrics(y_faces, y_pred_log, "Logistic Regression")

