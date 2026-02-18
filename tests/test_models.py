import torch
import os
import pytest
import sys

# Ajout du dossier src au path pour pouvoir importer les classes des modèles si nécessaire
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def test_pytorch_models_exist():
    """Vérifie que les modèles entraînés sont présents dans le dossier models."""
    models_to_check = ["models/torch_clf_model.pth", "models/torch_reg_model.pth"]
    for model_path in models_to_check:
        assert os.path.exists(model_path), f"Le modèle {model_path} est introuvable."


def test_model_loading_and_inference():
    """Teste si les modèles PyTorch peuvent être chargés sans erreur."""
    # Note: On charge via torch.load.
    # Pour un test plus complet, il faudrait importer la classe du modèle.
    clf_model_path = "models/torch_clf_model.pth"
    if os.path.exists(clf_model_path):
        try:
            model_data = torch.load(clf_model_path, map_location=torch.device("cpu"))
            assert model_data is not None
        except Exception as e:
            pytest.fail(f"Erreur lors du chargement du modèle de classification : {e}")
