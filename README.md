
# Course Completion Prediction & Student Performance Analysis

[![Python application](https://github.com/Gael926/dl-course-completion-prediction/actions/workflows/python-app.yml/badge.svg)](https://github.com/Gael926/dl-course-completion-prediction/actions/workflows/python-app.yml)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-yellow)

## Aperçu du Projet
Ce projet utilise des techniques de **Machine Learning** et **Deep Learning** pour analyser et prédire la réussite des étudiants dans un cours en ligne.

<p align="left">
  <a href="https://votre-app-streamlit.streamlit.app/">
    <img src="https://img.shields.io/badge/Live_Demo-Accéder_au_Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo" />
  </a>
</p>

<p align="left">
  <img src="reports/dashboard_demo.gif" width="90%" />
</p>
<p align="left"><i>Démo : Interface interactive de prédiction (Streamlit).</i></p>


---

## Résultats Clés (Test Set)

### Classification (Target: `Completed`)
Le modèle Deep Learning est optimisé pour maximiser le F1-Score sur la classe minoritaire.

<p align="left">
  <img src="https://raw.githubusercontent.com/Gael926/dl-course-completion-prediction/main/reports/figures/torch_class_loss.png" width="80%" />
</p>
<p align="left"><i>Courbe d'apprentissage : Convergence stable sans overfitting majeur.</i></p>

### Régression (Multi-output)
Un réseau de neurones unique prédit simultanément les 4 indicateurs de performance.

<p align="left">
  <img src="https://raw.githubusercontent.com/Gael926/dl-course-completion-prediction/main/reports/figures/torch_reg_loss.png" width="80%" />
</p>

#### Analyse des Prédictions : Réel vs Prédit
<p align="left">
  <img src="https://raw.githubusercontent.com/Gael926/dl-course-completion-prediction/main/reports/figures/torch_reg_predictions.png" width="90%" />
</p>
<p align="left"><i>Le modèle excelle sur le "Project Grade" (en haut à droite) mais peine sur la Satisfaction (en bas à gauche), qui semble aléatoire.</i></p>

#### Facteurs d'Influence (SHAP par Target)
<p align="left">
  <img src="https://raw.githubusercontent.com/Gael926/dl-course-completion-prediction/main/reports/figures/shap_regression_project_grade.png" width="45%" />
  <img src="https://raw.githubusercontent.com/Gael926/dl-course-completion-prediction/main/reports/figures/shap_regression_quiz_score_avg.png" width="45%" />
</p>
<p align="left">
  <img src="https://raw.githubusercontent.com/Gael926/dl-course-completion-prediction/main/reports/figures/shap_regression_time_spent_hours.png" width="45%" />
  <img src="https://raw.githubusercontent.com/Gael926/dl-course-completion-prediction/main/reports/figures/shap_regression_satisfaction_rating.png" width="45%" />
</p>
<p align="left"><i>Analyse fine : Chaque indicateur a ses propres drivers. Notez l'impact spécifique de l'éducation sur les Quiz.</i></p>

---

## Analyses Clés
*   **Engagement**: Les étudiants ayant complété le projet final ont 90% plus de chances de réussir le cours.
*   **Comportement**: Le temps passé sur le cours est moins prédictif que la performance aux tests intermédiaires.
*   **Robustesse**: Le Gradient Boosting reste le modèle de référence pour les données structurées de ce type.

### Difficultés & Analyse des Données
Certaines cibles (comme `Satisfaction` ou `Time_Spent`) sont extrêmement difficiles à prédire efficacement. Cela s'explique par le **manque de corrélation linéaire forte** dans les données, comme le montrent les matrices ci-dessous :

<p align="left">
  <img src="https://raw.githubusercontent.com/Gael926/dl-course-completion-prediction/main/reports/figures/correlation_target.png" width="45%" />
  <img src="https://raw.githubusercontent.com/Gael926/dl-course-completion-prediction/main/reports/figures/correlation_matrix_reg.png" width="45%" />
</p>
<p align="left"><i>Visualisation des corrélations : Impact direct sur la réussite (gauche) et Matrice de corrélation Régression (droite).</i></p>

---

## Installation & Quick Start

**Application en ligne** : [Accéder directement au Dashboard (Streamlit Cloud)](https://dl-course-completion-prediction.streamlit.app/)

### Installation
```bash
git clone https://github.com/Gael926/dl-course-completion-prediction.git
cd dl-course-completion-prediction
pip install -r requirements.txt
python -m streamlit run streamlit_app.py # Lancer le Dashboard (Streamlit)
```

### Commandes Avancées
*   **Pipeline ML Complet** : `python main.py` (Data Prep -> Entraînement -> Analyse)
*   **Tests Unitaires** : `pytest`
*   **API FastAPI** : `uvicorn app:app --reload`
*   **Docker** :
    ```bash
    docker build -t ml-course-prediction .
    docker run -p 8501:8501 ml-course-prediction
    ```

## Structure du Projet
```
├── data/                          # Données brutes et traitées
├── models/                        # Modèles (.pth) et Artefacts (.pkl)
│   ├── torch_clf_model.pth        # Poids du modèle de classification
│   ├── torch_reg_model.pth        # Poids du modèle de régression
│   └── *.pkl                      # Encoders et Scalers pour le préprocessing
├── notebooks/                     # Notebooks Jupyter (Exploration & Analyse)
│   ├── 00_data_prep.ipynb
│   ├── 01_baselines.ipynb
│   ├── 02_sklearn_models.ipynb
│   ├── 03_tf_models.ipynb
│   ├── 04_torch_models.ipynb
│   └── 05_model_analysis.ipynb
├── reports/figures/               # Graphiques générés pour le README
├── src/                           # Scripts Python modulaires (Production)
│   ├── 00_data_prep.py            # Nettoyage & Feature Engineering
│   ├── 01_baselines.py            # Modèles de base
│   ├── 02_sklearn_models.py       # Modèles Scikit-Learn
│   ├── 03_tf_models.py            # Implémentation TensorFlow
│   ├── 04_torch_models.py         # Implémentation PyTorch
│   ├── 05_model_analysis.py       # Analyse des résultats & SHAP
│   └── models_architectures.py    # Définition des classes PyTorch (NN)
├── tests/                         # Tests unitaires (Pytest)
│   ├── test_data.py               # Validation des données traitées
│   └── test_models.py             # Validation de l'intégrité des modèles
├── .github/workflows/             # Configuration GitHub Actions (CI/CD)
├── main.py                        # Script d'orchestration (Point d'entrée)
├── app.py                         # Serveur d'inférence FastAPI
├── streamlit_app.py               # Dashboard de visualisation
├── requirements.txt               # Dépendances du projet
├── Dockerfile                     # Configuration Docker
├── .dockerignore                  # Fichiers à ignorer par Docker
└── README.md                      # Documentation
```

## Impact Métier & Stratégie
- **Intervention Précoce** : Comme le `Project_Grade` est très prévisible, nous pouvons identifier tôt les étudiants à risque d'échec et leur proposer du tutorat.
- **Engagement** : Le temps passé (`Time_Spent`) n'est pas corrélé à la réussite dans ce dataset, suggérant que la *qualité* de l'étude prime sur la *quantité*.
