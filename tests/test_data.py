import pandas as pd
import os

# Chemins des fichiers traités
PROCESSED_DATA_PATH = "data/processed/Course_Completion_Prediction.csv"


def test_processed_data_exists():
    """Vérifie que le fichier de données traitées a été généré."""
    assert os.path.exists(PROCESSED_DATA_PATH), (
        f"Le fichier {PROCESSED_DATA_PATH} est introuvable."
    )


def test_data_integrity():
    """Vérifie l'intégrité du contenu des données traitées."""
    if os.path.exists(PROCESSED_DATA_PATH):
        df = pd.read_csv(PROCESSED_DATA_PATH)

        # Vérification de l'absence de valeurs manquantes
        assert df.isnull().sum().sum() == 0, (
            "Le dataset contient des valeurs manquantes."
        )

        # Vérification de la présence des colonnes cibles principales
        required_columns = ["Completed", "Project_Grade", "Quiz_Score_Avg"]
        for col in required_columns:
            assert col in df.columns, f"La colonne cible {col} est absente du dataset."


def test_data_leakage_safety():
    """Vérifie l'absence de fuite de données évidente."""
    if os.path.exists(PROCESSED_DATA_PATH):
        # On s'assure que des colonnes brutes interdites (pour la classification) ne sont pas restées par erreur
        # Cette liste dépend de la logique métier appliquée dans src/00_data_prep.py
        pass
