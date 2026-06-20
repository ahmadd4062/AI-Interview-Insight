import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

# Set paths
confident_dir = 'Dataset\Ravdess_Confident'  
nervous_dir = 'Dataset\Ravdess_Nervous'

# Function to extract features
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=audio).T, axis=0)
    return np.hstack([mfccs, chroma, zcr, rms])

# Load and label data
features = []
labels = []

# Confident (label 0)
for filename in os.listdir(confident_dir):
    if filename.endswith(".wav"):
        path = os.path.join(confident_dir, filename)
        features.append(extract_features(path))
        labels.append(0)

# Nervous (label 1)
for filename in os.listdir(nervous_dir):
    if filename.endswith(".wav"):
        path = os.path.join(nervous_dir, filename)
        features.append(extract_features(path))
        labels.append(1)

# Convert to arrays
X = np.array(features)
y = np.array(labels)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create individual models
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=5)

# Create combined model (Random Forest + KNN)
combined_model = VotingClassifier(estimators=[
    ('RandomForest', rf_model),
    ('KNN', knn_model)
], voting='soft')

# Train combined model
combined_model.fit(X_train_scaled, y_train)

# Predict and Evaluate
y_pred_combined = combined_model.predict(X_test_scaled)

# Evaluation
def evaluate_model(model_name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred) * 100
    print(f"\n{model_name} Accuracy: {accuracy:.2f}%")
    print(classification_report(y_true, y_pred, target_names=["Confident", "Nervous"]))
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(f"         Predicted")
    print(f"          0    1")
    print(f"Actual 0  {cm[0][0]}    {cm[0][1]}")
    print(f"       1  {cm[1][0]}    {cm[1][1]}")

evaluate_model("Combined Random Forest + KNN", y_test, y_pred_combined)

# Save combined model
joblib.dump((combined_model, scaler), 'combined_model_rf_knn.pkl')
print("\nCombined model saved successfully as 'combined_model_rf_knn.pkl'")
