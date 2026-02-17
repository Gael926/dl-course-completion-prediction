
# Course Completion Prediction & Student Performance Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-yellow)

## Aperçu du Projet
Ce projet utilise des techniques de **Machine Learning** et **Deep Learning** pour analyser et prédire la réussite des étudiants dans un cours en ligne.

### Analyse des Données

<p align="center">
  <img src="reports/figures/final_data_analysis.png" width="90%" />
</p>
<p align="center"><i>Visualisation des corrélations : Impact direct sur la réussite (gauche) et relations multi-variables (droite).</i></p>

---

## Résultats Clés (Test Set)

### Classification (Target: `Completed`)
Le problème est difficile (bruité), mais les modèles surpassent la baseline aléatoire.

<p align="center">
  <img src="reports/figures/final_accuracy_comparison.png" width="70%" />
</p>

| Modèle | Accuracy | F1-Score | Observations |
| :--- | :---: | :---: | :--- |
| **Baseline (Dummy)** | 49.87% | - | Performance aléatoire. |
| **Logistic Regression** | 56.40% | - | Simple mais efficace. |
| **Random Forest** | 59.40% | - | Capture les interactions complexes. |
| **Gradient Boosting** | **60.47%** | - | Très performant, capture des non-linéarités. |
| **PyTorch NN** | 60.02% | 0.61 | Bonnes performances, mais nécessite plus de tuning. |

<p align="center">
  <img src="reports/figures/final_classification.png" width="90%" />
</p>
<p align="center"><i>Performance du modèle Deep Learning : Matrice de confusion et impact des variables (SHAP).</i></p>

### Régression (Multi-output)
Nous avons utilisé des réseaux de neurones (PyTorch/TensorFlow) pour prédire les 4 variables simultanément.

<p align="center">
  <img src="reports/figures/sklearn_rmse_comparison.png" width="70%" />
</p>

| Target | RMSE (PyTorch) | R² | Interprétation |
| :--- | :---: | :---: | :--- |
| **Project Grade** | **3.56** | **0.94** | **Excellente prédiction**. Les features (quiz, activité) expliquent très bien la note finale. |
| **Quiz Score** | 12.31 | 0.04 | Difficile à prédire précisément uniquement via le comportement. |
| **Satisfaction** | 0.70 | ~0.00 | Aucune corrélation trouvée (probablement subjectif/aléatoire). |
| **Time Spent** | 3.82 | ~0.00 | Aucune corrélation trouvée avec les features disponibles. |

<p align="center">
  <img src="reports/figures/final_regression.png" width="90%" />
</p>
<p align="center"><i>Diagnostic de la régression : Prédictions sur les notes de projet et importance des features.</i></p>

---

## Analyses Clés
*   **Engagement**: Les étudiants ayant complété le projet final ont 90% plus de chances de réussir le cours.
*   **Comportement**: Le temps passé sur le cours est moins prédictif que la performance aux tests intermédiaires.
*   **Robustesse**: Le Gradient Boosting reste le modèle de référence pour les données structurées de ce type.

---

## Installation & Usage

1.  **Cloner le repo** :
    ```bash
    git clone https://github.com/votre-username/course-completion-prediction.git
    cd course-completion-prediction
    ```

2.  **Installer les dépendances** :
    ```bash
    pip install -r requirements.txt
    ```

3.  **Lancer le pipeline** :
    ```bash
    # 1. Préparer les données
    python src/00_data_prep.py

    # 2. Entraîner et évaluer les modèles
    python src/01_baselines.py
    python src/02_sklearn_models.py
    python src/04_torch_models.py
    ```

4.  **Explorer les Analyses** :
    Ouvrez `notebooks/05_model_analysis.ipynb` pour voir les **SHAP values** et l'importance des features.

## Structure du Projet
```
├── data/               # Données brutes et traitées
├── models/             # Modèles sauvegardés (.pth, .pkl)
├── notebooks/          # Notebooks Jupyter (Exploration & Analyse)
│   ├── 00_data_prep.ipynb
│   ├── 01_baselines.ipynb
│   ├── 02_sklearn_models.ipynb
│   ├── 03_tf_models.ipynb
│   ├── 04_torch_models.ipynb
│   └── 05_model_analysis.ipynb
├── reports/            # Rapports et visualisations
│   └── figures/        # Graphiques générés pour le README
├── src/                # Scripts Python modulaires (Production)
│   ├── 00_data_prep.py        # Nettoyage & Feature Engineering
│   ├── 01_baselines.py        # Modèles de base
│   ├── 02_sklearn_models.py   # Modèles Scikit-Learn
│   ├── 03_tf_models.py        # Implémentation TensorFlow
│   ├── 04_torch_models.py     # Implémentation PyTorch
│   ├── 05_model_analysis.py   # Analyse des résultats & SHAP
├── requirements.txt    # Dépendances du projet
└── README.md           # Documentation
```

## Impact Métier & Stratégie
- **Intervention Précoce** : Comme le `Project_Grade` est très prévisible, nous pouvons identifier tôt les étudiants à risque d'échec et leur proposer du tutorat.
- **Engagement** : Le temps passé (`Time_Spent`) n'est pas corrélé à la réussite dans ce dataset, suggérant que la *qualité* de l'étude prime sur la *quantité*.
