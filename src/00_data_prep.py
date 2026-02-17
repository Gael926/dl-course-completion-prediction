# %% [markdown]
# # Exploratory Data Analysis (EDA)
#
# ## Dataset: Student Course Completion Prediction
# This notebook performs a comprehensive EDA to understand the data, identify patterns, and prepare for modeling.

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

# Configuration
sns.set_style("whitegrid")
pd.set_option("display.max_columns", None)

# %% [markdown]
# ## 1. Data Loading and Overview

# %%
file_path = "../data/raw/Course_Completion_Prediction.csv"
df = pd.read_csv(file_path)
# on garde que les categorie = a programming
df = df[df["Category"] == "Programming"]
print(f"Dataset Shape: {df.shape}")
df.head()

# %% [markdown]
# ### Feature Descriptions
#
# | Feature | Description |
# | :--- | :--- |
# | Student_ID | Unique identifier for the student. |
# | Name | Full name of the student. |
# | Gender | Gender of the student (e.g., Male, Female, Others). |
# | Age | Age of the student in years. |
# | Education_Level | Highest level of education attained by the student. |
# | Employment_Status | Current employment status of the student (e.g., Student, Employed). |
# | City | City where the student resides. |
# | Device_Type | Type of device used to access the course (e.g., Laptop, Mobile). |
# | Internet_Connection_Quality | Quality of the student's internet connection (e.g., Low, Medium, High). |
# | Course_ID | Unique identifier for the course. |
# | Course_Name | Full name of the course. |
# | Category | Subject category of the course (e.g., Programming, Design). |
# | Course_Level | Difficulty level of the course (e.g., Beginner, Advanced). |
# | Course_Duration_Days | Total intended duration of the course in days. |
# | Instructor_Rating | Average rating given to the course instructor (e.g., 1.0 to 5.0). |
# | Login_Frequency | Number of times the student logs in per a defined period (e.g., per week). |
# | Average_Session_Duration_Min | Average time (in minutes) spent per login session. |
# | Video_Completion_Rate | Percentage of course video content watched by the student. |
# | Discussion_Participation | Count of the student's participation in discussion forums. |
# | Time_Spent_Hours | Total number of hours spent actively engaging with the course content. |
# | Days_Since_Last_Login | Number of days passed since the student's most recent login. |
# | Notifications_Checked | Count of how many times the student checked their notifications. |
# | Peer_Interaction_Score | A score representing the quality or quantity of interaction with peers. |
# | Assignments_Submitted | Total count of assignments successfully submitted by the student. |
# | Assignments_Missed | Total count of assignments missed or not submitted by the student. |
# | Quiz_Attempts | Total number of attempts made across all course quizzes. |
# | Quiz_Score_Avg | Average score (percentage) across all quizzes taken by the student. |
# | Project_Grade | The final grade received on the course project (e.g., 0 to 100). |
# | Progress_Percentage | Overall completion progress in the course (0 to 100). |
# | Rewatch_Count | Total number of times the student rewatched course video content. |
# | Enrollment_Date | Date when the student initially enrolled in the course. |
# | Payment_Mode | Method used for course fee payment (e.g., Credit Card, Scholarship). |
# | Fee_Paid | Binary indicator: whether the course fee was paid ('Yes'/'No'). |
# | Discount_Used | Binary indicator: whether a discount or coupon was applied ('Yes'/'No'). |
# | Payment_Amount | The final amount paid by the student for the course. |
# | App_Usage_Percentage | Percentage of course activity conducted via the mobile application. |
# | Reminder_Emails_Clicked | Number of course reminder emails the student clicked open. |
# | Support_Tickets_Raised | Total number of support or help tickets raised by the student. |
# | Satisfaction_Rating | Overall satisfaction rating given by the student (e.g., 1.0 to 5.0). |
# | Completed | Target Variable: Binary indicator if the student completed the course. |

# %%
df.info()

# %%
df[["Quiz_Score_Avg", "Project_Grade"]].corr()

# %%
df.describe()

# %% [markdown]
# ## 2. Data Cleaning & Preprocessing
# *   Check for missing values
# *   Check for duplicates
# *   Convert dates
# *   Drop irrelevant columns

# %%
df.isnull().sum()

# %% [markdown]
# > Aucune valeur nulle

# %%
# Duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate Rows: {duplicates}")

# %% [markdown]
# > Aucune ligne dupliquée

# %%
# trouver les features numérique et catégorielle
numerical_cols = []
categorical_cols = []
for col in df.columns:
    if df[col].dtype == "object":
        categorical_cols.append(col)
    else:
        numerical_cols.append(col)
print(numerical_cols)
print(categorical_cols)

# %% [markdown]
# ### Encodage

# %%
# Définition des listes de colonnes à encoder
cardinal_label_encode = ["City", "Course_Name"]
categorical_label_encode = [
    "Education_Level",
    "Course_Level",
    "Internet_Connection_Quality",
]
categorical_one_hot_encode = [
    "Gender",
    "Device_Type",
    "Employment_Status",
    "Payment_Mode",
    "Fee_Paid",
    "Discount_Used",
]

# %%
# Label Encoding pour les colonnes à haute cardinalité (City, Course_Name)
# On utilise LabelEncoder car ces colonnes ont beaucoup de valeurs uniques
label_encoder = LabelEncoder()
for col in cardinal_label_encode:
    df[col] = label_encoder.fit_transform(df[col])

print(f"City: {df['City'].nunique()} valeurs uniques")
print(f"Course_Name: {df['Course_Name'].nunique()} valeurs uniques")

# %%
# Ordinal Encoding pour les colonnes catégorielles ordonnées

# Définition des ordres pour chaque colonne
education_order = ["HighSchool", "Diploma", "Bachelor", "Master", "PhD"]
course_level_order = ["Beginner", "Intermediate", "Advanced"]
internet_order = ["Low", "Medium", "High"]

ordinal_encoder = OrdinalEncoder(
    categories=[education_order, course_level_order, internet_order]
)
df[categorical_label_encode] = ordinal_encoder.fit_transform(
    df[categorical_label_encode]
)

# %%
# One-Hot Encoding pour les colonnes catégorielles nominales

df = pd.get_dummies(df, columns=categorical_one_hot_encode, drop_first=True)

print(f"\nNouvelle forme du dataset: {df.shape}")
print("\nNouvelles colonnes créées:")
print([col for col in df.columns if any(x in col for x in categorical_one_hot_encode)])

# %%
# Encodage de la variable cible pour la classification
# Completed: 'Completed' -> 1, 'Not Completed' -> 0

df["Completed"] = df["Completed"].map({"Completed": 1, "Not Completed": 0})

print("Encodage de la variable cible 'Completed':")
print(df["Completed"].value_counts())

# %%
# Vérification du dataset encodé
print("Dataset après encodage:")
print(df.dtypes)
print(f"\nShape: {df.shape}")
df.head()

# %% [markdown]
# ### Split des deux datasets (Classification et Regression) + Normalisation des données numériques

# %%
# Définition des colonnes administrative à supprimer systématiquement
excluded_cols = ["Course_ID", "Student_ID", "Category", "Name", "Enrollment_Date"]

# DATASET POUR LA REGRESSION
target_cols_reg = [
    "Quiz_Score_Avg",
    "Project_Grade",
    "Satisfaction_Rating",
    "Time_Spent_Hours",
]
df_reg = df.drop(
    columns=[
        col
        for col in excluded_cols + ["Completed", "Quiz_Attempts"]
        if col in df.columns
    ]
)

X_reg = df_reg.drop(columns=[col for col in target_cols_reg if col in df_reg.columns])
y_reg = df_reg[[col for col in target_cols_reg if col in df_reg.columns]]

# Scaling de X_reg
scaler_reg = StandardScaler()
num_cols_reg = X_reg.select_dtypes(
    include=["number", "bool", "int64", "float64"]
).columns
X_reg[num_cols_reg] = scaler_reg.fit_transform(X_reg[num_cols_reg])

print(f"Regression: X_reg scaled ({X_reg.shape}), y_reg intact ({y_reg.shape})")

# DATASET POUR LA CLASSIFICATION
target_col_class = "Completed"
# Pour la classification, on garde Quiz_Score_Avg et Project_Grade comme features
df_class = df.drop(columns=[col for col in excluded_cols if col in df.columns])

X_class = df_class.drop(columns=[target_col_class])
y_class = df_class[target_col_class]

# Scaling de X_class
scaler_class = StandardScaler()
num_cols_class = X_class.select_dtypes(
    include=["number", "bool", "int64", "float64"]
).columns
X_class[num_cols_class] = scaler_class.fit_transform(X_class[num_cols_class])

print(
    f"Classification: X_class scaled ({X_class.shape}), y_class intact ({y_class.shape})"
)


# %%
# On concatène X et y pour calculer la corrélation globale
df_class_corr = pd.concat([X_class, y_class], axis=1)

# Calcul de la corrélation avec la cible uniquement
corr_with_target = df_class_corr.corr()["Completed"].sort_values(ascending=False)

# Affichage visuel
plt.figure(figsize=(10, 12))
sns.heatmap(corr_with_target.to_frame(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Corrélation des Features avec la cible 'Completed'")
plt.tight_layout()
plt.savefig("../reports/figures/correlation_target.png", dpi=300)
plt.show()

# Affichage texte des 10 plus importantes
print("Top 10 des variables les plus corrélées à 'Completed' :")
print(corr_with_target.head(10))

# %%
# Liste de vos 4 targets
targets_reg = [
    "Quiz_Score_Avg",
    "Project_Grade",
    "Satisfaction_Rating",
    "Time_Spent_Hours",
]

# On concatène X et y
df_reg_corr = pd.concat([X_reg, y_reg], axis=1)

# Calcul de la matrice de corrélation
# On filtre pour ne garder que la corrélation entre les Features (X) et les Targets (y)
corr_matrix_reg = df_reg_corr.corr().loc[X_reg.columns, targets_reg]

# Affichage visuel (Heatmap)
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix_reg, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Corrélation entre les Features et les 4 Targets de Régression")
plt.tight_layout()
plt.savefig("../reports/figures/correlation_matrix_reg.png", dpi=300)
plt.show()

# %% [markdown]
# ---
#
# ### Export des données

# %%
# SAUVEGARDE DES DATASETS DANS data/processed
import os

# Créer le dossier processed s'il n'existe pas
processed_path = "../data/processed"
os.makedirs(processed_path, exist_ok=True)

# Sauvegarde des datasets de régression
X_reg.to_csv(f"{processed_path}/X_regression.csv", index=False)
y_reg.to_csv(f"{processed_path}/y_regression.csv", index=False)

# Sauvegarde des datasets de classification
X_class.to_csv(f"{processed_path}/X_classification.csv", index=False)
y_class.to_csv(f"{processed_path}/y_classification.csv", index=False)

print("Datasets sauvegardés dans data/processed/")
print(f"- X_regression.csv ({X_reg.shape})")
print(f"- y_regression.csv ({y_reg.shape})")
print(f"- X_classification.csv ({X_class.shape})")
print(f"- y_classification.csv ({y_class.shape})")
