import subprocess
import sys
import os


def run_script(script_path):
    """Exécute un script Python et vérifie son code de retour."""
    print(f"[*] Exécution de : {script_path}...")
    # On utilise sys.executable pour garantir qu'on utilise le même interpréteur
    result = subprocess.run(
        [sys.executable, script_path], capture_output=False, text=True
    )

    if result.returncode != 0:
        print(f"[!] Erreur lors de l'exécution de {script_path}")
        # On ne sort pas forcément du script si une étape échoue (ex: baselines),
        # mais pour les données (00) c'est critique.
        if "00_data_prep.py" in script_path:
            sys.exit(1)
    else:
        print(f"[+] {script_path} terminé avec succès.\n")


def main():
    # Définir les étapes du pipeline dans l'ordre chronologique
    scripts = [
        "src/00_data_prep.py",  # Préparation des données
        "src/01_baselines.py",  # Modèles de base
        "src/02_sklearn_models.py",  # Modèles Scikit-Learn (RF, GBM)
        "src/03_tf_models.py",  # Modèles TensorFlow (Optionnel)
        "src/04_torch_models.py",  # Modèles PyTorch (Production)
        "src/05_model_analysis.py",  # Analyse SHAP & Interprétation
    ]

    # S'assurer que les dossiers de sortie existent
    for folder in ["data/processed", "models", "reports/figures"]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"[*] Création du dossier : {folder}")

    print("\nDÉMARRAGE DU PIPELINE DE PRÉDICTION\n")

    for script in scripts:
        if os.path.exists(script):
            run_script(script)
        else:
            print(f"[?] Script ignoré (non trouvé) : {script}")

    print("\nTOUT EST PRÊT !")
    print("Les modèles sont dans 'models/' et les images dans 'reports/figures/'.\n")


if __name__ == "__main__":
    main()
