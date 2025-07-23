import subprocess

try:
    subprocess.run(["python3", "-m", "src.preprocessing"], check=True)
    subprocess.run(["python3", "-m", "src.models"], check=True)
    subprocess.run(["python3", "-m", "src.train"], check=True)
    subprocess.run(["python3", "-m", "src.predict"], check=True)
except subprocess.CalledProcessError as e:
    print(f"Une erreur est survenue : {e}")