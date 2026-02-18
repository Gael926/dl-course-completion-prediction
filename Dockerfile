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

# Exposer le port par défaut de Streamlit
EXPOSE 8501

# Commande par défaut : Lancer le dashboard Streamlit
# On utilise python -m streamlit pour éviter les problèmes de PATH
CMD ["python", "-m", "streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
