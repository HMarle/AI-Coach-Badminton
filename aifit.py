import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split, learning_curve
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import joblib

# Chemin vers le dossier contenant les fichiers CSV
folder_path = '/Users/hadrienmarle/Desktop/wetransfer_defensif_2024-06-04_1231/fichiers_coupé'
folder_path_secure = '/Users/hadrienmarle/Desktop/sécure'

# Fonction pour charger les données
def load_data(folder_path):
    data = []
    max_length = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            # Extraire l'étiquette avant le premier point
            label = filename.split('_')[0]
            filepath = os.path.join(folder_path, filename)
            # Lire le fichier CSV
            df = pd.read_csv(filepath)
            # Convertir les valeurs en flottants et gérer les valeurs séparées par des points-virgules
            features = []
            for value in df.values.flatten():
                if isinstance(value, str):
                    features.extend([float(x) for x in value.split(';')])
                else:
                    features.append(float(value))
            # Ajuster la longueur maximale
            if len(features) > max_length:
                max_length = len(features)
            # Ajouter les caractéristiques et l'étiquette à la liste de données
            data.append({"features": features, "label": label})

    return data, max_length

# Charger les données
data, max_length = load_data(folder_path)
data_secure, max_length_secure = load_data(folder_path_secure)

# Afficher le nombre total de fichiers chargés
print(f"Total files loaded: {len(data)}")

# Trouver la longueur maximale de caractéristiques entre les deux ensembles de données
final_max_length = max(max_length, max_length_secure)

# Remplir ou tronquer les listes de caractéristiques pour qu'elles aient la même longueur
def adjust_features(data, max_length):
    for item in data:
        if len(item["features"]) < max_length:
            item["features"].extend([0.0] * (max_length - len(item["features"])))
        elif len(item["features"]) > max_length:
            item["features"] = item["features"][:max_length]
    return data

# Ajuster les caractéristiques des deux ensembles de données
data = adjust_features(data, final_max_length)
data_secure = adjust_features(data_secure, final_max_length)

# Convertir les données en DataFrame
data_df = pd.DataFrame(data)
data_secure_df = pd.DataFrame(data_secure)

# Séparer les caractéristiques (X) et l'étiquette (y)
X = np.array(data_df['features'].tolist())
y = np.array(data_df['label'].tolist())

# Séparer les caractéristiques (X) et l'étiquette (y) pour les données sécurisées
X_secure = np.array(data_secure_df['features'].tolist())
y_secure = np.array(data_secure_df['label'].tolist())

# Afficher la taille des ensembles de données
print(f"Total samples in dataset: {X.shape[0]}")
print(f"Total samples in secure dataset: {X_secure.shape[0]}")

# Vérifier la cohérence du nombre de caractéristiques
if X.shape[1] != X_secure.shape[1]:
    raise ValueError(f"Feature mismatch: {X.shape[1]} features in dataset but {X_secure.shape[1]} features in secure dataset")

# Standardiser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_secure_scaled = scaler.transform(X_secure)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Afficher la taille des ensembles d'entraînement et de test
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Définir les paramètres de la recherche par grille
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

# Créer un modèle SVC avec GridSearchCV
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5)
grid.fit(X_train, y_train)

# Afficher les meilleurs paramètres
print(f"Best parameters: {grid.best_params_}")

# Prédire les étiquettes pour l'ensemble de test
y_pred = grid.predict(X_test)

# Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Fonction pour tracer la matrice de confusion
def plot_confusion_matrix(cm, labels, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# Afficher la matrice de confusion
plot_confusion_matrix(conf_matrix, labels=np.unique(y_test))

# Rapport de classification
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)

# Prédire les étiquettes pour les données sécurisées
model = grid.best_estimator_
y_secure_pred = model.predict(X_secure_scaled)
print('Test secure predictions:', y_secure_pred)

# Courbes d'apprentissage
train_sizes, train_scores, test_scores = learning_curve(grid.best_estimator_, X_scaled, y, cv=5)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

##Sauvegarder
save_directory = '/Users/hadrienmarle/Desktop/model'

# Créer le dossier s'il n'existe pas déjà
os.makedirs(save_directory, exist_ok=True)

# Sauvegarder le modèle et le scaler dans ce dossier
model_path = os.path.join(save_directory, 'svm_model.joblib')
scaler_path = os.path.join(save_directory, 'scaler.joblib')

joblib.dump(grid.best_estimator_, model_path)
joblib.dump(scaler, scaler_path)
# Tracer les courbes d'apprentissage
plt.figure()
plt.plot(train_sizes, train_scores_mean, 'o-', label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', label="Cross-validation score")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.legend(loc="best")
plt.title("Learning Curves")
plt.show()

