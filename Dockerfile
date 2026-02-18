# Utiliser une image Python légère
FROM python:3.10-slim

# Éviter la génération de fichiers .pyc et activer le buffering des logs
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances système nécessaires (si besoin)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copier le fichier des dépendances
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du code du projet
COPY . .

# Créer les dossiers nécessaires s'ils n'existent pas
RUN mkdir -p data/processed models reports/figures

# Commande par défaut : Lancer le pipeline complet
CMD ["python", "main.py"]
