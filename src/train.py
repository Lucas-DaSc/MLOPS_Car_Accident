import os
import joblib
import pandas as pd
import glob

from src.preprocessing import scale_and_encode
from src.models import get_model_sgdc 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import mlflow
import mlflow.sklearn

# Import data & preprocess 
def data_split(path):
    df = pd.read_csv(path, sep=',')
    df = scale_and_encode(df)

    # Split
    X = df.drop('Accident_Severity', axis=1)
    y = df['Accident_Severity']

    assert X.shape[1] == 19

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    return X_train, X_test, y_train, y_test


def train_and_log_model(model, model_name, batch_num, X_train, y_train, X_test, y_test):
    # Entraînement incrémental
    if batch_num == 1:
        model.partial_fit(X_train, y_train, classes=[0,1,2,3]) 
    else:
        model.partial_fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Démarrer un run MLflow
    with mlflow.start_run(run_name=f"{model_name}_batch_{batch_num}"):

        # Log des paramètres et métriques
        mlflow.log_param("batch_number", batch_num)
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", acc)

        # Exemple d'entrée
        input_example = X_test.iloc[:1]

        # Log du modèle en tant qu'artefact
        logged_model_info = mlflow.sklearn.log_model(
            sk_model=model,
            input_example=input_example,
            name="model"
        )

        # Enregistrement dans le Model Registry
        mlflow.register_model(
            model_uri=logged_model_info.model_uri,
            name=model_name
        )

        # Sauvegarde locale
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, f"models/{model_name}_v{batch_num}.pkl")

        print(f"[Batch {batch_num}] Accuracy: {acc:.4f} - Version MLflow créée")

    return model


if __name__ == "__main__":
    # Dossier contenant les batchs
    batch_folder = "/home/lucas/MLOPS_Test/data_cleaned/"
    batch_files = sorted(glob.glob(os.path.join(batch_folder, "data_clean_*.csv")))
    
    # Experiment
    mlflow.set_experiment("car_accident_severity_fix")

    # Initialiser le modèle SGD
    model = get_model_sgdc()
    model_name = 'SGDC'

    for i, batch_path in enumerate(batch_files, start=1):
        print(f"\n🔁 Traitement du batch {i}: {batch_path}")
        X_train, X_test, y_train, y_test = data_split(batch_path)
        model = train_and_log_model(
            model=model,
            model_name=model_name,
            batch_num=i,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )