import pandas as pd
import os 

def split(csv_path, dossier_sortie="sorties", prefixe_fichier="dataset_part"):
    # Chargement du dataset
    df = pd.read_csv(csv_path)
    
    # Création du dossier de sortie s'il n'existe pas
    os.makedirs(dossier_sortie, exist_ok=True)

    # Taille de chaque partie
    taille = len(df) // 10
    reste = len(df) % 10

    debut = 0
    for i in range(10):
        fin = debut + taille + (1 if i < reste else 0)
        partie = df.iloc[debut:fin]
        fichier_sortie = os.path.join(dossier_sortie, f"{prefixe_fichier}_{i+1}.csv")
        partie.to_csv(fichier_sortie, index=False)
        debut = fin

    print(f"Dataset divisé et sauvegardé dans le dossier '{dossier_sortie}'.")

split('/home/<user>/.cache/kagglehub/datasets/atharvasoundankar/road-accidents-dataset/versions/1/Road_Accident_Data.csv')