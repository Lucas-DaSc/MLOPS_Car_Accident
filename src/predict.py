import os
import glob
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from src.train import data_split
import mlflow.pyfunc

# Import data & preprocess
X_train, X_test, y_train, y_test = data_split('/home/lucas/MLOPS_Test/data_cleaned/data_clean_1.csv')

def evaluate(model, X_test, y_test, batch_name=""):
    preds = model.predict(X_test)

    # Conversion si nécessaire
    if isinstance(preds, (pd.DataFrame, pd.Series)):
        preds = preds.values.ravel()

    acc = accuracy_score(y_test, preds)
    print(f"[{batch_name}] Accuracy: {acc:.4f}")
    return acc

def load_model_from_mlflow(model_name, run_id=None):
    if run_id:
        model_uri = f"runs:/{run_id}/{model_name}"
    else:
        model_uri = f"models:/{model_name}/Production"
    return mlflow.pyfunc.load_model(model_uri)


def load_model():
    try:
        model = load_model_from_mlflow("SGDC_model")
        print("✅ SGDC model loaded from MLflow registry")
        return model
    except Exception as e:
        print(f"⚠️ Erreur MLflow: {e}")
        print("📂 Fallback: chargement depuis fichier local")
        return joblib.load("models/SGDC_v1.pkl")


def get_batch_files(data_dir):
    pattern = os.path.join(data_dir, "data_clean_*.csv")
    return sorted(glob.glob(pattern))


if __name__ == "__main__":
    data_dir = "/home/lucas/MLOPS_Test/data_cleaned"
    batch_files = get_batch_files(data_dir)

    print(f"\n🔎 {len(batch_files)} batchs trouvés dans {data_dir}.")

    if not batch_files:
        print("❌ Aucun fichier batch trouvé.")
        exit(1)

    model = load_model()
    results = {}

    for batch_file in batch_files:
        batch_name = os.path.basename(batch_file)
        print(f"\n📦 Traitement du batch : {batch_name}")
        try:
            X_train, X_test, y_train, y_test = data_split(batch_file)

            # Vérification que les datasets ne sont pas vides
            if X_test.empty or y_test.empty:
                raise ValueError("X_test ou y_test est vide.")

            acc = evaluate(model, X_test, y_test, batch_name)
            results[batch_name] = acc

        except Exception as e:
            print(f"❌ Erreur lors du traitement de {batch_name}: {e}")
            traceback.print_exc()  # Affiche le stacktrace complet
            results[batch_name] = None
        finally:
            print(f"✅ Fin du traitement de {batch_name}")
