import streamlit as st
import torch
import joblib
import pandas as pd
from src.models_architectures import (
    CourseCompletionClassifier,
    StudentPerformanceRegressor,
)

# Configuration de la page
st.set_page_config(page_title="Course Prediction Dashboard", layout="wide")

# Custom CSS pour le Light Mode
st.markdown(
    """
    <style>
    /* Améliorations spécifiques au Light Mode via Media Query native */
    @media (prefers-color-scheme: light) {
        .main {
            background-color: #fcfcfc;
        }
        div[data-testid="stMetricValue"] {
            color: #1a1a1a;
            font-weight: 700;
        }
        div[data-testid="stExpander"], .stContainer {
            background-color: white !important;
            border: 1px solid #e0e0e0 !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
        }
        h1, h2, h3 {
            color: #111111 !important;
        }
    }
    
    /* Styles globaux pour le relief */
    div[data-testid="stForm"] {
        border-radius: 10px;
        padding: 20px;
    }
    .stButton>button {
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Titre principal centré
st.markdown(
    "<h1 style='text-align: center;'>Course Completion & Performance Prediction</h1>",
    unsafe_allow_html=True,
)
st.markdown("---")


# Chargement des assets
@st.cache_resource
def load_assets():
    # Encoders & Columns
    label_encoders = joblib.load("models/label_encoders.pkl")
    ordinal_encoder = joblib.load("models/ordinal_encoder.pkl")
    X_class_columns = joblib.load("models/X_class_columns.pkl")
    X_reg_columns = joblib.load("models/X_reg_columns.pkl")

    # Scalers
    scaler_class = joblib.load("models/scaler_class.pkl")
    scaler_reg = joblib.load("models/scaler_reg.pkl")

    # Modèles
    model_class = CourseCompletionClassifier(input_dim=len(X_class_columns))
    model_class.load_state_dict(
        torch.load("models/torch_clf_model.pth", map_location="cpu")
    )
    model_class.eval()

    model_reg = StudentPerformanceRegressor(input_dim=len(X_reg_columns), output_dim=4)
    model_reg.load_state_dict(
        torch.load("models/torch_reg_model.pth", map_location="cpu")
    )
    model_reg.eval()

    return {
        "le": label_encoders,
        "oe": ordinal_encoder,
        "cols_class": X_class_columns,
        "cols_reg": X_reg_columns,
        "scaler_class": scaler_class,
        "scaler_reg": scaler_reg,
        "model_class": model_class,
        "model_reg": model_reg,
    }


assets = load_assets()


# logique d'inférence
def preprocess_input(data_dict, columns_ref):
    df = pd.DataFrame([data_dict])
    # Encodage Ordinal
    cat_ordinal = ["Education_Level", "Course_Level", "Internet_Connection_Quality"]
    df[cat_ordinal] = assets["oe"].transform(df[cat_ordinal])
    # Encodage Label
    for col, le in assets["le"].items():
        if col in df.columns:
            try:
                df[col] = le.transform(df[col])
            except ValueError:
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
    # Alignement
    return df.reindex(columns=columns_ref, fill_value=0)


# données par défaut
default_data = {
    "Student_ID": "S001",
    "Name": "Etudiant Test",
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
    "Time_Spent_Hours": 20.0,
    "Days_Since_Last_Login": 2,
    "Notifications_Checked": 5,
    "Peer_Interaction_Score": 8.5,
    "Assignments_Submitted": 4,
    "Assignments_Missed": 0,
    "Quiz_Attempts": 2,
    "Quiz_Score_Avg": 85.0,
    "Project_Grade": 90.0,
    "Progress_Percentage": 60.0,
    "Rewatch_Count": 3,
    "Enrollment_Date": "2023-01-01",
    "Payment_Mode": "Credit Card",
    "Fee_Paid": "Yes",
    "Discount_Used": "No",
    "Payment_Amount": 199.99,
    "App_Usage_Percentage": 40.0,
    "Reminder_Emails_Clicked": 2,
    "Support_Tickets_Raised": 0,
    "Satisfaction_Rating": 4.8,
}

# formulaire
st.subheader("Profil Etudiant")
with st.container(border=True):
    r1_c1, r1_c2, r1_c3 = st.columns(3)
    with r1_c1:
        age = st.number_input("Age", 15, 60, default_data["Age"], step=1)
    with r1_c2:
        login_freq = st.number_input(
            "Frequence Connexion/semaine", 0, 7, default_data["Login_Frequency"]
        )
    with r1_c3:
        gender_fr = st.selectbox("Genre", ["Homme", "Femme"], index=0)
        gender = "Male" if gender_fr == "Homme" else "Female"

    r2_c1, r2_c2 = st.columns(2)
    with r2_c1:
        name = st.text_input("Nom", default_data["Name"])
    with r2_c2:
        education = st.selectbox(
            "Niveau d'Etude",
            ["HighSchool", "Diploma", "Bachelor", "Master", "PhD"],
            index=2,
        )

    r3_c1, r3_c2, r3_c3, r3_c4 = st.columns(4)
    with r3_c1:
        video_rate = st.slider(
            "Videos Completees (%)", 0.0, 1.0, default_data["Video_Completion_Rate"]
        )
    with r3_c2:
        quiz_avg = st.slider(
            "Moyenne Quiz (%)", 0.0, 100.0, default_data["Quiz_Score_Avg"]
        )
    with r3_c3:
        project_grade = st.slider(
            "Note Projet (%)", 0.0, 100.0, default_data["Project_Grade"]
        )
    with r3_c4:
        satisfaction = st.slider(
            "Satisfaction (1-5)", 1.0, 5.0, default_data["Satisfaction_Rating"]
        )

st.markdown("<br><br>", unsafe_allow_html=True)
_, center_col, _ = st.columns([2, 2, 2])
with center_col:
    submitted = st.button(
        "Lancer la prediction", type="primary", use_container_width=True
    )

st.markdown("<br><br><hr>", unsafe_allow_html=True)

# calculs et affichage
if submitted:
    # Préparation du payload local
    payload = default_data.copy()
    payload.update(
        {
            "Name": name,
            "Age": age,
            "Gender": gender,
            "Education_Level": education,
            "Login_Frequency": login_freq,
            "Video_Completion_Rate": video_rate,
            "Quiz_Score_Avg": quiz_avg,
            "Project_Grade": project_grade,
            "Satisfaction_Rating": satisfaction,
        }
    )

    col1, col2 = st.columns(2)

    with col1:
        st.header("Classification")
        st.subheader("Course Completion")
        df_aligned = preprocess_input(payload, assets["cols_class"])
        X_scaled = assets["scaler_class"].transform(df_aligned)
        with torch.no_grad():
            prob = assets["model_class"](torch.FloatTensor(X_scaled)).item()
            is_completed = prob > 0.5

        inner_col, _ = st.columns([3, 1])
        with inner_col:
            st.metric(label="Probabilite de Reussite", value=f"{prob * 100:.1f} %")
            st.progress(prob)
            if is_completed:
                st.success("SUCCES: L'étudiant devrait terminer le cours.")
            else:
                st.error("RISQUE: Abandon probable détecté.")

    with col2:
        st.header("Regression")
        st.subheader("Multi-Outputs pour les Indicateurs de Performance")
        df_aligned_reg = preprocess_input(payload, assets["cols_reg"])
        X_scaled_reg = assets["scaler_reg"].transform(df_aligned_reg)
        with torch.no_grad():
            preds = assets["model_reg"](torch.FloatTensor(X_scaled_reg)).numpy()[0]

        results_reg = {
            "Quiz_Score_Avg": round(float(preds[0]), 2),
            "Project_Grade": round(float(preds[1]), 2),
            "Satisfaction_Rating": round(float(preds[2]), 2),
            "Time_Spent_Hours": round(float(preds[3]), 2),
        }

        c_a, c_b = st.columns(2)
        c_a.metric("Quiz Score Avg (Prédit)", f"{results_reg['Quiz_Score_Avg']:.1f}")
        c_b.metric("Project Grade (Prédit)", f"{results_reg['Project_Grade']:.1f}")
        c_c, c_d = st.columns(2)
        c_c.metric(
            "Satisfaction (Prédite)", f"{results_reg['Satisfaction_Rating']:.1f}/5"
        )
        c_d.metric("Temps Passe (Prédit)", f"{results_reg['Time_Spent_Hours']:.1f} h")
        st.bar_chart(results_reg)
else:
    st.info("Utilisez le formulaire ci-dessus pour simuler un profil étudiant.")
