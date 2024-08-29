import cv2
import mediapipe as mp
import os
import numpy as np
import csv
import pandas as pd

# Initialiser Mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Chemin vers le dossier contenant les vidéos
video_folder = '/Users/hadrienmarle/Desktop/wetransfer_defensif_2024-06-04_1231/vidéos/defensif'
# Dossier de destination des fichiers CSV
dossier_destination = "/Users/hadrienmarle/Desktop/wetransfer_defensif_2024-06-04_1231/données"
# Vérifier si le dossier de destination existe, sinon le créer
os.makedirs(dossier_destination, exist_ok=True)

# Liste tous les fichiers dans le dossier
video_files = [f for f in os.listdir(video_folder) if os.path.isfile(os.path.join(video_folder, f))]

# Parcourir chaque vidéo
for video_index, video_file in enumerate(video_files, start=1):
    video_path = os.path.join(video_folder, video_file)
    cap = cv2.VideoCapture(video_path)

    # Initialisation des listes pour cette vidéo
    data_LS_x, data_LS_y = [], []
    data_RS_x, data_RS_y = [], []
    data_LE_x, data_LE_y = [], []
    data_RE_x, data_RE_y = [], []
    data_RW_x, data_RW_y = [], []
    data_LW_x, data_LW_y = [], []

    if not cap.isOpened():
        print(f"Erreur lors de l'ouverture de la vidéo {video_path}")
        continue

    # Définir les landmarks et les connexions pour le bras droit et l'épaule gauche
    arm_landmarks = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_WRIST,
        mp_pose.PoseLandmark.LEFT_WRIST,
        mp_pose.PoseLandmark.LEFT_ELBOW
    ]
    right_arm_connections = [
        (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
        (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST)
    ]

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Fin de la vidéo ou erreur de lecture.")
                break

            # Traitement de l'image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                # Collecter les coordonnées de l'épaule gauche
                left_shoulder_point = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                data_LS_x.append(left_shoulder_point.x)
                data_LS_y.append(left_shoulder_point.y)

                # Collecter les coordonnées de l'épaule droite
                right_shoulder_point = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                data_RS_x.append(right_shoulder_point.x)
                data_RS_y.append(right_shoulder_point.y)

                # Collecter les coordonnées du coude gauche
                left_elbow_point = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                data_LE_x.append(left_elbow_point.x)
                data_LE_y.append(left_elbow_point.y)

                # Collecter les coordonnées du coude droit
                right_elbow_point = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                data_RE_x.append(right_elbow_point.x)
                data_RE_y.append(right_elbow_point.y)

                # Collecter les coordonnées du poignet droit
                right_wrist_point = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                data_RW_x.append(right_wrist_point.x)
                data_RW_y.append(right_wrist_point.y)

                # Collecter les coordonnées du poignet gauche
                left_wrist_point = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value]
                data_LW_x.append(left_wrist_point.x)
                data_LW_y.append(left_wrist_point.y)

                # Tracer les connexions et les landmarks
                for connection in right_arm_connections:
                    start_idx = connection[0].value
                    end_idx = connection[1].value
                    start_point = results.pose_landmarks.landmark[start_idx]
                    end_point = results.pose_landmarks.landmark[end_idx]
                    start_coords = (int(start_point.x * image.shape[1]), int(start_point.y * image.shape[0]))
                    end_coords = (int(end_point.x * image.shape[1]), int(end_point.y * image.shape[0]))
                    cv2.line(image, start_coords, end_coords, (255, 255, 255), 2)

                for landmark in arm_landmarks:
                    landmark_point = results.pose_landmarks.landmark[landmark.value]
                    landmark_coords = (int(landmark_point.x * image.shape[1]), int(landmark_point.y * image.shape[0]))
                    cv2.circle(image, landmark_coords, 5, (0, 0, 255), -1)

            # Afficher l'image
            cv2.imshow("Pose Detection", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Maintenant, 'q' sert à quitter le script complètement
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Normalisation
    normalized_LS_x = (np.array(data_LS_x) - np.min(data_LS_x)) / (np.max(data_LS_x) - np.min(data_LS_x))
    normalized_LS_y = (np.array(data_LS_y) - np.min(data_LS_y)) / (np.max(data_LS_y) - np.min(data_LS_y))
    normalized_RS_x = (np.array(data_RS_x) - np.min(data_RS_x)) / (np.max(data_RS_x) - np.min(data_RS_x))
    normalized_RS_y = (np.array(data_RS_y) - np.min(data_RS_y)) / (np.max(data_RS_y) - np.min(data_RS_y))

    normalized_LE_x = (np.array(data_LE_x) - np.min(data_LE_x)) / (np.max(data_LE_x) - np.min(data_LE_x))
    normalized_LE_y = (np.array(data_LE_y) - np.min(data_LE_y)) / (np.max(data_LE_y) - np.min(data_LE_y))
    normalized_RE_x = (np.array(data_RE_x) - np.min(data_RE_x)) / (np.max(data_RE_x) - np.min(data_RE_x))
    normalized_RE_y = (np.array(data_RE_y) - np.min(data_RE_y)) / (np.max(data_RE_y) - np.min(data_RE_y))

    normalized_RW_x = (np.array(data_RW_x) - np.min(data_RW_x)) / (np.max(data_RW_x) - np.min(data_RW_x))
    normalized_RW_y = (np.array(data_RW_y) - np.min(data_RW_y)) / (np.max(data_RW_y) - np.min(data_RW_y))
    normalized_LW_x = (np.array(data_LW_x) - np.min(data_LW_x)) / (np.max(data_LW_x) - np.min(data_LW_x))
    normalized_LW_y = (np.array(data_LW_y) - np.min(data_LW_y)) / (np.max(data_LW_y) - np.min(data_LW_y))

    # Données de la vidéo actuelle
    video_data = {
        'LS_x': normalized_LS_x,
        'LS_y': normalized_LS_y,
        'RS_x': normalized_RS_x,
        'RS_y': normalized_RS_y,

        'LE_x': normalized_LE_x,
        'LE_y': normalized_LE_y,
        'RE_x': normalized_RE_x,
        'RE_y': normalized_RE_y,

        'RW_x': normalized_RW_x,
        'RW_y': normalized_RW_y,
        'LW_x': normalized_LW_x,
        'LW_y': normalized_LW_y
    }

    # Générer un nom de fichier unique avec l'index de la vidéo
    nom_fichier = f"defensif_video_{video_index}.csv"
    chemin_fichier = os.path.join(dossier_destination, nom_fichier)

    maxvalues = max(len(normalized_LS_x), len(normalized_RS_x), len(normalized_LE_x), len(normalized_RE_x), len(normalized_RW_x), len(normalized_LW_x))

    # Écriture des données dans le fichier CSV
    with open(chemin_fichier, mode='w', newline='') as fichier:
        writer = csv.writer(fichier)
        writer.writerow(['LS_x', 'LS_y', 'RS_x', 'RS_y', 'LE_x', 'LE_y', 'RE_x', 'RE_y', 'RW_x', 'RW_y', 'LW_x', 'LW_y'])  # En-tête
        for i in range(maxvalues):
            writer.writerow([
                normalized_LS_x[i] if i < len(normalized_LS_x) else '',
                normalized_LS_y[i] if i < len(normalized_LS_y) else '',
                normalized_RS_x[i] if i < len(normalized_RS_x) else '',
                normalized_RS_y[i] if i < len(normalized_RS_y) else '',
                normalized_LE_x[i] if i < len(normalized_LE_x) else '',
                normalized_LE_y[i] if i < len(normalized_LE_y) else '',
                normalized_RE_x[i] if i < len(normalized_RE_x) else '',
                normalized_RE_y[i] if i < len(normalized_RE_y) else '',
                normalized_RW_x[i] if i < len(normalized_RW_x) else '',
                normalized_RW_y[i] if i < len(normalized_RW_y) else '',
                normalized_LW_x[i] if i < len(normalized_LW_x) else '',
                normalized_LW_y[i] if i < len(normalized_LW_y) else ''
            ])

    print("Fichier CSV créé avec succès pour la vidéo", video_index, ":", chemin_fichier)
