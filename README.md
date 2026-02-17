
# Course Completion Prediction & Student Performance Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-yellow)

## Aperçu du Projet
Ce projet utilise des techniques de **Machine Learning** et **Deep Learning** pour analyser et prédire la réussite des étudiants dans un cours en ligne.

### Analyse des Données


<p align="center">
  <img src="reports/figures/correlation_combined.png" width="90%" />
</p>
*Gauche : Corrélation avec la réussite (Completed). Droite : Corrélations entre features et cibles de régression.*

**Objectifs :**
1.  **Classification** : Prédire si un étudient va compléter le cours (`Completed`: 0 ou 1).
2.  **Régression (Multi-output)** : Estimer simultanément 4 indicateurs de performance :
    -   `Project_Grade` (Note du projet final)
    -   `Quiz_Score_Avg` (Moyenne des quiz)
    -   `Satisfaction_Rating` (Niveau de satisfaction)
    -   `Time_Spent_Hours` (Temps passé)

## Résultats Clés (Test Set)

### Classification (Target: `Completed`)
Le problème est difficile (bruité), mais les modèles surpassent la baseline aléatoire.

<p align="center">
  <img src="reports/figures/sklearn_accuracy_comparison.png" width="60%" />
</p>

| Modèle | Accuracy | F1-Score | Observations |
| :--- | :---: | :---: | :--- |
| **Baseline (Dummy)** | 49.87% | - | Performance aléatoire. |
| **Logistic Regression** | **60.84%** | - | Modèle le plus simple et robuste. |
| **Gradient Boosting** | **60.47%** | - | Très performant, capture des non-linéarités. |
| **PyTorch NN** | 60.02% | 0.61 | Bonnes performances, mais nécessite plus de tuning. |


<table>
  <tr>
    <td align="center" width="50%"><img src="reports/figures/torch_confusion_matrix.png" width="100%" /></td>
    <td align="center" width="50%"><img src="reports/figures/shap_classification_summary.png" width="100%" /></td>
  </tr>
  <tr>
    <td align="center"><i>Matrice de Confusion (PyTorch)</i></td>
    <td align="center"><i>Impact SHAP des features</i></td>
  </tr>
</table>

### Régression (Multi-output)
Nous avons utilisé des réseaux de neurones (PyTorch/TensorFlow) pour prédire les 4 variables simultanément.

<p align="center">
  <img src="reports/figures/sklearn_rmse_comparison.png" width="60%" />
</p>

| Target | RMSE (PyTorch) | R² | Interprétation |
| :--- | :---: | :---: | :--- |
| **Project Grade** | **3.56** | **0.94** | 🌟 **Excellente prédiction**. Les features (quiz, activité) expliquent très bien la note finale. |
| **Quiz Score Avg** | 12.37 | 0.03 | Difficile à prédire avec les données actuelles. |
| **Satisfaction** | 0.70 | ~0.00 | Aucune corrélation trouvée (probablement subjectif/aléatoire). |
| **Time Spent** | 3.82 | ~0.00 | Aucune corrélation trouvée avec les features disponibles. |


<table>
  <tr>
    <td align="center" width="50%"><img src="reports/figures/torch_reg_predictions.png" width="100%" /></td>
    <td align="center" width="50%"><img src="reports/figures/shap_regression_project_grade.png" width="100%" /></td>
  </tr>
  <tr>
    <td align="center"><i>Prédictions vs Réel (Project Grade)</i></td>
    <td align="center"><i>Impact SHAP sur la note du projet</i></td>
  </tr>
</table>

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
├── data/               # Données brutes et procesées
├── models/             # Modèles sauvegardés (.pth, .pkl)
├── notebooks/          # Notebooks Jupyter pour l'analyse (EDA, SHAP)
├── src/                # Scripts Python modulaires
│   ├── 00_data_prep.py        # Nettoyage & Feature Engineering
│   ├── 01_baselines.py        # Modèles simples
│   ├── 02_sklearn_models.py   # Random Forest, GBM
│   ├── 03_tf_models.py        # TensorFlow Implementation
│   └── 04_torch_models.py     # PyTorch Implementation (Production)
├── requirements.txt    # Dépendances
└── README.md           # Documentation du projet (ce fichier)
```

## Insights Business
- **Intervention Précoce** : Comme le `Project_Grade` est très prévisible, nous pouvons identifier tôt les étudiants à risque d'échec et leur proposer du tutorat.
- **Engagement** : Le temps passé (`Time_Spent`) n'est pas corrélé à la réussite dans ce dataset, suggérant que la *qualité* de l'étude prime sur la *quantité*.
