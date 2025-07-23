🚦 Projet MLOps – Sécurité Routière
Ce projet a pour objectif d'entraîner et de monitorer un modèle simple de machine learning à partir de différents batchs de données issues d'un dataset sur la sécurité routière.

L'accent est mis sur l'application des bonnes pratiques MLOps afin de garantir un cycle de vie robuste, reproductible et maintenable. Il constitue un exemple pédagogique pour la mise en œuvre concrète de l'approche MLOps dans un contexte réel.

🏗️ Architecture du projet
mlops_car_accident/
├── src/                    # Code source
│   ├── data/
│   ├── features/
│   ├── models/
│   └── monitoring/
├── main.py              # Fichier principale
├── extract_data.py         # Extraction données
├── requirements.txt        # Dépendances Python
└── README.md               # Description du projet

⚙️ Fonctionnalités
Chargement des données batchées
Entraînement progressif d’un modèle simple (classification)
Tracking des expériences et des métriques
Monitoring continu de la performance
Structure modulaire pour faciliter les tests et évolutions
Utilisation d’outils MLOps (MLflow)
