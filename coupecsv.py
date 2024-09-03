import os
import pandas as pd

# Chemin vers le dossier contenant les fichiers CSV
dossier_chemin = ''
end_path = ''

# Obtenir la liste des fichiers CSV dans le dossier
fichiers_csv = [f for f in os.listdir(dossier_chemin) if f.endswith('.csv')]

# Lire les fichiers et stocker leurs DataFrames dans une liste
dataframes = []
fichiers_a_supprimer = []

for fichier in fichiers_csv:
    fichier_chemin = os.path.join(dossier_chemin, fichier)
    if os.path.getsize(fichier_chemin) >= 8 * 1024:  # Vérifier si le fichier fait 8 ko ou plus
        df = pd.read_csv(fichier_chemin)
        dataframes.append(df)
    else:
        fichiers_a_supprimer.append(fichier)  # Ajouter le fichier à la liste des fichiers à supprimer

# Supprimer les fichiers CSV trop petits
for fichier in fichiers_a_supprimer:
    os.remove(os.path.join(dossier_chemin, fichier))

# Vérifier s'il y a des fichiers suffisants pour traitement
if dataframes:
    # Trouver le nombre minimum de lignes parmi tous les fichiers
    min_lignes = min([df.shape[0] for df in dataframes])

    # Créer un nouveau dossier pour les fichiers modifiés
    dossier_modifie_chemin = os.path.join(end_path, "fichiers_coupé")
    os.makedirs(dossier_modifie_chemin, exist_ok=True)

    # Traiter chaque fichier pour conserver seulement le nombre minimum de lignes au milieu
    for i, df in enumerate(dataframes):
        total_lignes = df.shape[0]
        lignes_a_supprimer = total_lignes - min_lignes
        debut = lignes_a_supprimer // 2
        fin = total_lignes - (lignes_a_supprimer - debut)

        df_tronque = df.iloc[debut:fin]

        # Sauvegarder le DataFrame tronqué dans un nouveau fichier CSV
        fichier_modifie_chemin = os.path.join(dossier_modifie_chemin, fichiers_csv[i])
        df_tronque.to_csv(fichier_modifie_chemin, index=False)

    print("Tous les fichiers ont été traités et sauvegardés dans le dossier 'fichiers_coupé'. Les fichiers CSV trop petits ont été supprimés.")
else:
    print("Aucun fichier CSV suffisant trouvé pour traitement.")
