from fastapi import FastAPI
from pydantic import BaseModel
import torch
import joblib
import pandas as pd

# importation des architectures partagées
from src.models_architectures import (
    CourseCompletionClassifier,
    StudentPerformanceRegressor,
)

# Chargement des artefacts de préprocessing
label_encoders = joblib.load("models/label_encoders.pkl")
ordinal_encoder = joblib.load("models/ordinal_encoder.pkl")
X_class_columns = joblib.load("models/X_class_columns.pkl")
X_reg_columns = joblib.load("models/X_reg_columns.pkl")


# Schémas de données pydantic (Strings pour les catégories)
class BaseInput(BaseModel):
    Student_ID: str
    Name: str
    Gender: str
    Age: int
    Education_Level: str
    Employment_Status: str
    City: str
    Device_Type: str
    Internet_Connection_Quality: str
    Course_ID: str
    Course_Name: str
    Category: str
    Course_Level: str
    Course_Duration_Days: int
    Instructor_Rating: float
    Login_Frequency: int
    Average_Session_Duration_Min: float
    Video_Completion_Rate: float
    Discussion_Participation: int
    Days_Since_Last_Login: int
    Notifications_Checked: int
    Peer_Interaction_Score: float
    Assignments_Submitted: int
    Assignments_Missed: int
    Progress_Percentage: float
    Rewatch_Count: int
    Enrollment_Date: str
    Payment_Mode: str
    Fee_Paid: str
    Discount_Used: str
    Payment_Amount: float
    App_Usage_Percentage: float
    Reminder_Emails_Clicked: int
    Support_Tickets_Raised: int

    model_config = {
        "json_schema_extra": {
            "example": {
                "Student_ID": "S001",
                "Name": "Jean Dupont",
                "Gender": "Male",
                "Age": 25,
                "Education_Level": "Bachelor",
                "Employment_Status": "Student",
                "City": "New York",
                "Device_Type": "Laptop",
                "Internet_Connection_Quality": "High",
                "Course_ID": "C001",
                "Course_Name": "Python For Beginners",
                "Category": "Programming",
                "Course_Level": "Beginner",
                "Course_Duration_Days": 45,
                "Instructor_Rating": 4.5,
                "Login_Frequency": 5,
                "Average_Session_Duration_Min": 30.5,
                "Video_Completion_Rate": 0.8,
                "Discussion_Participation": 10,
                "Days_Since_Last_Login": 2,
                "Notifications_Checked": 5,
                "Peer_Interaction_Score": 8.5,
                "Assignments_Submitted": 4,
                "Assignments_Missed": 0,
                "Progress_Percentage": 0.6,
                "Rewatch_Count": 3,
                "Enrollment_Date": "2023-01-01",
                "Payment_Mode": "Credit Card",
                "Fee_Paid": "Yes",
                "Discount_Used": "No",
                "Payment_Amount": 199.99,
                "App_Usage_Percentage": 0.4,
                "Reminder_Emails_Clicked": 2,
                "Support_Tickets_Raised": 0,
                "Quiz_Attempts": 2,
                "Quiz_Score_Avg": 85.0,
                "Project_Grade": 90.0,
                "Time_Spent_Hours": 20.0,
                "Satisfaction_Rating": 4.8,
            }
        }
    }


class ClassificationInput(BaseInput):
    Quiz_Attempts: int
    Quiz_Score_Avg: float
    Project_Grade: float
    Time_Spent_Hours: float
    Satisfaction_Rating: float


class RegressionInput(BaseInput):
    # Pour la régression, on ne demande pas les targets (Quiz_Score, Project_Grade, etc.)
    # Ni Quiz_Attempts qui est exclu
    pass


# Fonction utilitaire de préprocessing
def preprocess_input(data_dict, columns_ref):
    df = pd.DataFrame([data_dict])

    # Encodage Ordinal
    cat_ordinal = ["Education_Level", "Course_Level", "Internet_Connection_Quality"]
    # Vérification que les valeurs sont valides, sinon fallback ou erreur gérée par FastAPI
    df[cat_ordinal] = ordinal_encoder.transform(df[cat_ordinal])

    # Encodage Label (City, Course_Name)
    # Attention: Si une valeur est nouvelle, LabelEncoder plantera.
    # Pour la production, on devrait gérer les 'unknown'. Ici on espère que les entrées matchent le train set.
    for col, le in label_encoders.items():
        if col in df.columns:
            # Astuce pour gérer les inconnus: assigner une classe majoritaire ou -1
            # Ici on fait simple: on assume que l'input est valide
            # Pour éviter le crash:
            try:
                df[col] = le.transform(df[col])
            except ValueError:
                # Fallback: utiliser la classe 0 (souvent arbitraire mais évite le crash)
                df[col] = 0

    # One-Hot Encoding
    cat_ohe = [
        "Gender",
        "Device_Type",
        "Employment_Status",
        "Payment_Mode",
        "Fee_Paid",
        "Discount_Used",
    ]
    df = pd.get_dummies(df, columns=cat_ohe, drop_first=True)

    # Alignement des colonnes (Reindexing)
    # Ajoute les colonnes manquantes (fill_value=0) et retire les colonnes en trop
    df_aligned = df.reindex(columns=columns_ref, fill_value=0)

    return df_aligned


# Chargement des assets
app = FastAPI(title="Course Completion API")

# Chargement des scalers
scaler_class = joblib.load("models/scaler_class.pkl")
scaler_reg = joblib.load("models/scaler_reg.pkl")

# Chargement des modèles
INPUT_DIM_CLASS = len(X_class_columns)  # Devrait être 42
model_class = CourseCompletionClassifier(input_dim=INPUT_DIM_CLASS)
model_class.load_state_dict(
    torch.load("models/torch_clf_model.pth", map_location="cpu")
)
model_class.eval()

INPUT_DIM_REG = len(X_reg_columns)  # Devrait être 37
model_reg = StudentPerformanceRegressor(input_dim=INPUT_DIM_REG, output_dim=4)
model_reg.load_state_dict(torch.load("models/torch_reg_model.pth", map_location="cpu"))
model_reg.eval()

# Endpoints


@app.post("/predict/completion")
def predict_completion(data: ClassificationInput):
    # Préprocessing robuste
    df_aligned = preprocess_input(data.dict(), X_class_columns)

    # Normalisation
    X_scaled = scaler_class.transform(df_aligned)
    X_tensor = torch.FloatTensor(X_scaled)

    # Inférence
    with torch.no_grad():
        prob = model_class(X_tensor).item()
        prediction = 1 if prob > 0.5 else 0

    return {"is_completed": prediction, "probability": round(prob, 4)}


@app.post("/predict/performance")
def predict_performance(data: RegressionInput):
    # Préprocessing robuste
    df_aligned = preprocess_input(data.dict(), X_reg_columns)

    # Normalisation
    X_scaled = scaler_reg.transform(df_aligned)
    X_tensor = torch.FloatTensor(X_scaled)

    # Inférence
    with torch.no_grad():
        preds = model_reg(X_tensor).numpy()[0]

    return {
        "Quiz_Score_Avg": round(float(preds[0]), 2),
        "Project_Grade": round(float(preds[1]), 2),
        "Satisfaction_Rating": round(float(preds[2]), 2),
        "Time_Spent_Hours": round(float(preds[3]), 2),
    }


@app.get("/")
def home():
    return {"message": "API de prédiction active. Allez sur /docs."}


if __name__ == "__main__":
    import uvicorn
    import socket

    def find_available_port(start_port, max_attempts=10):
        for port in range(start_port, start_port + max_attempts):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("0.0.0.0", port))
                    return port
                except OSError:
                    continue
        return start_port

    port = find_available_port(8000)
    print(f"Lancement de l'API sur le port : {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
