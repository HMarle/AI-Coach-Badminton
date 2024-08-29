import cv2
import mediapipe as mp
import numpy as np
import joblib
import os

# Chemin vers le répertoire où les modèles sont sauvegardés
load_directory = '/Users/hadrienmarle/Desktop/TIPE 2/model'
model_path = os.path.join(load_directory, 'svm_model.joblib')
scaler_path = os.path.join(load_directory, 'scaler.joblib')

# Vérifiez si les fichiers existent
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

# Charger le modèle et le scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Initialiser MediaPipe pour la détection des poses
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Fonction pour extraire les caractéristiques de la pose
def extract_pose_landmarks(results):
    landmarks = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            landmarks.append(landmark.x)
            landmarks.append(landmark.y)
    return landmarks

# Fonction pour ajuster la longueur des caractéristiques
def adjust_features_length(features, max_length):
    if len(features) < max_length:
        features.extend([0.0] * (max_length - len(features)))
    elif len(features) > max_length:
        features = features[:max_length]
    return features

# Chemin vers la vidéo
video_path = '/Users/hadrienmarle/Desktop/TIPE 2/test/2.mov'  # Remplacez par le chemin correct de votre vidéo

# Vérifiez que le fichier vidéo existe
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video file not found: {video_path}")

# Ouvrir la vidéo
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Erreur : Impossible de lire la vidéo à partir de {video_path}")

# Variables pour le stockage des caractéristiques
features_list = []
final_max_length = 444  # 33 points (x, y) pour la détection 2D avec MediaPipe

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir l'image en RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processus MediaPipe
    results = pose.process(image)

    # Extraire les caractéristiques de la pose
    features = extract_pose_landmarks(results)
    if len(features) > 0:
        # Ajuster les caractéristiques à la longueur maximale
        adjusted_features = adjust_features_length(features, final_max_length)
        features_list.append(adjusted_features)

        # Convertir les caractéristiques en NumPy array et normaliser
        X_new = np.array([adjusted_features])
        X_new_scaled = scaler.transform(X_new)

        # Faire des prédictions
        prediction = model.predict(X_new_scaled)[0]

        # Afficher les résultats MediaPipe
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )

        # Ajouter le texte de la prédiction sur la frame
        label = 'Offensif' if prediction == 'offensif' else 'defensif'
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Afficher la frame
    cv2.imshow('MediaPipe Pose', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
