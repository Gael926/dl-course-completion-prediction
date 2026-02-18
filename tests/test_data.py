import pandas as pd
import os

# Chemins des fichiers traités
PROCESSED_FILES = [
    "data/processed/X_classification.csv",
    "data/processed/y_classification.csv",
    "data/processed/X_regression.csv",
    "data/processed/y_regression.csv",
]


def test_processed_data_exists():
    """Vérifie que les fichiers de données traitées ont été générés."""
    for file_path in PROCESSED_FILES:
        assert os.path.exists(file_path), f"Le fichier {file_path} est introuvable."


def test_data_integrity():
    """Vérifie l'intégrité du contenu des fichiers de classification."""
    x_path = "data/processed/X_classification.csv"
    y_path = "data/processed/y_classification.csv"

    if os.path.exists(x_path) and os.path.exists(y_path):
        df_x = pd.read_csv(x_path)
        df_y = pd.read_csv(y_path)

        # Vérification de l'absence de valeurs manquantes
        assert df_x.isnull().sum().sum() == 0, (
            "X_classification contient des valeurs manquantes."
        )
        assert df_y.isnull().sum().sum() == 0, (
            "y_classification contient des valeurs manquantes."
        )

        # Vérification de la présence des colonnes cibles principales (dans X_classification car elles y sont scalées)
        # Mais "Completed" est dans y_classification
        required_columns_x = ["Project_Grade", "Quiz_Score_Avg"]
        for col in required_columns_x:
            assert col in df_x.columns, (
                f"La colonne cible {col} est absente de X_classification."
            )

        assert df_y.shape[0] == df_x.shape[0], "Mismatch de lignes entre X et y."


def test_data_leakage_safety():
    """Vérifie l'absence de fuite de données évidente."""
    x_path = "data/processed/X_classification.csv"
    if os.path.exists(x_path):
        df_x = pd.read_csv(x_path)
        # On s'assure que des colonnes brutes interdites ne sont pas restées
        forbidden = ["Course_ID", "Student_ID", "Category", "Name", "Enrollment_Date"]
        for col in forbidden:
            assert col not in df_x.columns, f"La colonne interdite {col} est présente."
