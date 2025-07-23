import pandas as pd
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Import datas
def load_data_batches(folder_path):
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist or is not a directory.")

    files = sorted(folder.glob("dataset_part_*.csv"))
    if not files:
        raise FileNotFoundError(f"No files matching 'dataset_part_*.csv' found in '{folder_path}'.")

    return [(f.name, pd.read_csv(f)) for f in files]

# Preprocessing 

#Date
def date_conversion(df, colonne_date):
    # Check existence of the column
    assert colonne_date in df.columns, f"Column '{colonne_date}' not found in DataFrame"

    # Backup original null count
    n_missing_before = df[colonne_date].isna().sum()

    # Conversion to datetime
    df[colonne_date] = pd.to_datetime(df[colonne_date],format="%d-%m-%Y", errors='coerce').dt.strftime('%B')

    # Post-conversion check
    n_missing_after = df[colonne_date].isna().sum()

    assert n_missing_after <= n_missing_before, (
        f"Conversion to datetime introduced new NaNs: {n_missing_before} → {n_missing_after}"
    )

    return df


# Week and Week-End
def week_weekend(df,colonne_day):
    # Conversion 
    mapping = {
        'Monday': 'Week',
        'Tuesday': 'Week',
        'Wednesday': 'Week',
        'Thursday': 'Week',
        'Friday': 'Week',
        'Saturday': 'Week-End',
        'Sunday': 'Week-End'
    }

    df[colonne_day] = df[colonne_day].replace(mapping)
    assert set(df[colonne_day].unique()) == {'Week', 'Week-End'}

    return df


#Visibility
def visibility(df,colonne_visibility):
    # Conversion 
    mapping = {
       'Daylight':'Good Visibility',
       'Darkness - lights lit' : 'Good Visibility',
       'Darkness - lighting unknown' : 'Bad Visibility',
       'Darkness - lights unlit' : 'Bad Visibility',
       'Darkness - no lighting' : 'Bad Visibility'
       }

    df[colonne_visibility] = df[colonne_visibility].replace(mapping)
    assert set(df[colonne_visibility].unique()) == {'Good Visibility', 'Bad Visibility'}
    
    return df

#Moment of the day
import logging
logging.basicConfig(filename='error_log.txt', level=logging.WARNING)

def hour_moment(df, colonne_heure, nouvelle_colonne):
    
    # Retirer les NaN
    df = df.dropna(subset=[colonne_heure]).copy()

    # Convertir en datetime.time
    heures = pd.to_datetime(df[colonne_heure], format="%H:%M", errors='coerce').dt.hour
    heures_valides = heures.dropna()
    assert heures_valides.between(0, 23).all()
    
    # Définir les moments de la journée
    valeurs = ['Night', 'Morning', 'Afternoon', 'Evening']

    # Créer la nouvelle colonne
    df.loc[:, nouvelle_colonne] = pd.cut(heures, bins=[-1,5,11,17,23], labels=valeurs).astype('object')

    #Missing 
    n_missing_before = df[colonne_heure].isna().sum()
    n_missing_after = df[nouvelle_colonne].isna().sum()
    
    assert n_missing_after <= n_missing_before, (
        f"Conversion to datetime introduced new NaNs: {n_missing_before} → {n_missing_after}")

    valeurs_unique = set(df[nouvelle_colonne].unique())
    assert valeurs_unique == set(valeurs)

    #Enlever l'ancienne colonne
    df = df.drop(colonne_heure, axis=1)
    assert colonne_heure not in df.columns

    return df


# Road Surface
def road_surface(df, colonne_road):
    # Conversion
    mapping = {
       'Dry':'Good Surface',
       'Wet or damp' : 'Bad Surface',
       'Frost or ice' : 'Bad Surface',
       'Snow' : 'Bad Surface',
       'Flood over 3cm. deep' : 'Bad Surface'
       }

    df[colonne_road] = df[colonne_road].replace(mapping)
    assert set(df[colonne_road].unique()) == {'Good Surface', 'Bad Surface'}

    return df

# Weather

def weather(df, colonne_weather):
    # Conversion
    mapping = {
       'Fine no high winds':'No Bad Weather',
       'Raining no high winds' : 'Bad Weather',
       'Snowing no high winds' : 'Bad Weather',
       'Fine + high winds' : 'Bad Weather',
       'Fog or mist' : 'Very Bad Weather',
       'Raining + high winds' : 'Very Bad Weather',
       'Snowing + high winds' : 'Very Bad Weather'
       }

    df[colonne_weather] = df[colonne_weather].replace(mapping)
    assert set(df[colonne_weather].unique()) == {'No Bad Weather', 'Bad Weather', 'Very Bad Weather','Other'}

    return df

# Vehicle

def vehicle(df, colonne_vehicle):
    # Conversion
    mapping = {
       'Car':'Light Vehicle',
       'Taxi/Private hire car' : 'Light Vehicle',
       'Van / Goods 3.5 tonnes mgw or under' : 'Light Vehicle',
       'Motorcycle over 500cc' : 'Motorcycle',
       'Motorcycle 125cc and under' : 'Motorcycle',
       'Motorcycle 50cc and under' : 'Motorcycle',
       'Motorcycle over 125cc and up to 500cc' : 'Motorcycle',
       'Goods over 3.5t. and under 7.5t' : 'Large Vehicle',
       'Bus or coach (17 or more pass seats)' : 'Large Vehicle',
       'Goods 7.5 tonnes mgw and over' : 'Large Vehicle',
       'Agricultural vehicle' : 'Large Vehicle',
       'Minibus (8 - 16 passenger seats)' : 'Large Vehicle',
       'Pedal cycle' : 'No Vehicle', 
       'Ridden horse': 'No Vehicle'
       }

    df[colonne_vehicle] = df[colonne_vehicle].replace(mapping)
    assert set(df[colonne_vehicle].unique()) == {'Light Vehicle', 'Motorcycle', 'Large Vehicle','Other vehicle', 'No Vehicle'}

    return df

def number_casualties(df, colonne):
    
    df[colonne] = df[colonne].apply(lambda x: 'Classe 1' if x < 3 else 'Classe 2')
    assert set(df[colonne].unique()) == {'Classe 1','Classe 2'}
    return df

# Global
def preprocess_batch(df):
    df = df.drop(columns='Accident_Index', errors='ignore')

    df = date_conversion(df, colonne_date="Accident Date")
    df = week_weekend(df, colonne_day='Day_of_Week')
    df = visibility(df, colonne_visibility='Light_Conditions')
    df = hour_moment(df, colonne_heure='Time', nouvelle_colonne='Moment_Day')
    df = road_surface(df, colonne_road='Road_Surface_Conditions')
    df = weather(df, colonne_weather='Weather_Conditions')
    df = vehicle(df, colonne_vehicle='Vehicle_Type')
    df = number_casualties(df, colonne='Number_of_Casualties')

    return df

# Sauvegarder data cleaned
def save_clean_data(df, dossier_sortie, nom_fichier):
    if df is None:
        raise ValueError('Dataframe est None')
    os.makedirs(dossier_sortie, exist_ok=True)
    chemin_fichier = os.path.join(dossier_sortie, nom_fichier)
    df.to_csv(chemin_fichier, index=False)

# Standardisation & Encodage

def scale_and_encode(df):
    #Répartition
    numerical_columns = df.select_dtypes(exclude='object')
    categorical_columns = df.select_dtypes(include='object')

    numerical_columns_name = df.select_dtypes(exclude='object').columns
    categorical_columns_name = df.select_dtypes(include='object').columns

    #Standardisation
    scaler = StandardScaler()
    df[numerical_columns_name] = scaler.fit_transform(df[numerical_columns_name])

    for col in numerical_columns_name:
        assert pd.api.types.is_numeric_dtype(df[col])

    # Encoder les colonnes catégorielles
    encoder = LabelEncoder()
    for col in categorical_columns:
        df[col] = encoder.fit_transform(df[col])
        assert pd.api.types.is_numeric_dtype(df[col])

    return df

if __name__ == "__main__":
    input_folder = '/home/lucas/MLOPS_Test/sorties'
    output_folder = '/home/lucas/MLOPS_Test/data_cleaned'

    batches = load_data_batches(input_folder)
    print(f"Nombres de batchs : {len(batches)}")

    for i, (filename, df) in enumerate(batches, start=1):
        print(f"🔁 Traitement du batch {i} : {filename}")

        try:
            df_clean = preprocess_batch(df)

        except Exception as e:
            print(f"❌ Erreur dans le batch {i} ({filename}) : {e}")
            continue  # On passe au batch suivant

        if df_clean.shape[1] != 20:
            print(f"⚠️ Attention : le batch {i} ({filename}) a {df_clean.shape[1]} colonnes au lieu de 20")

        output_file = f"data_clean_{i}.csv"
        save_clean_data(df_clean, dossier_sortie=output_folder, nom_fichier=output_file)
        print(f"✅ Fichier enregistré : {output_file}")