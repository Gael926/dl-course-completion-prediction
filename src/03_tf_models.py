# %% [markdown]
# # Modèles TensorFlow - Classification et Régression
# 
# Ce notebook implémente deux modèles avec TensorFlow/Keras :
# - **Classification** : Prédire si un étudiant va compléter le cours (`Completed`).
# - **Régression** : Prédire les scores et la satisfaction (`Quiz_Score_Avg`, `Project_Grade`, `Satisfaction_Rating`, `Time_Spent_Hours`).

# %%
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, Activation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. Chargement des données

# %%
# Chargement des fichiers exportés
X_class = pd.read_csv('../data/processed/X_classification.csv')
y_class = pd.read_csv('../data/processed/y_classification.csv')

X_reg = pd.read_csv('../data/processed/X_regression.csv')
y_reg = pd.read_csv('../data/processed/y_regression.csv')

print("Datasets chargés successfully.")
print(f"Classification : X={X_class.shape}, y={y_class.shape}")
print(f"Régression : X={X_reg.shape}, y={y_reg.shape}")

# %% [markdown]
# ## 2. PARTIE 1 : CLASSIFICATION
# ---
# 
# Objectif : Prédire la colonne `Completed` (0 ou 1).

# %%
# Split des données
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class.values, y_class.values.ravel(), test_size=0.15, random_state=42
)

# %%
# Création du modèle de classification (simplifié pour éviter l'overfitting)
model_class = Sequential([
    Input(shape=(X_train_c.shape[1],)),
    
    Dense(128),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.4),
    
    Dense(64),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),
    
    Dense(32),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.2),
    
    Dense(1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model_class.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model_class.summary()

# %%
# Callbacks pour stabiliser l'apprentissage
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=30, # Arrêt si aucune amélioration après 30 époques
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,   # Divise le LR par 5 si on stagne
        patience=10,  # Après 10 époques sans progrès
        min_lr=1e-6,
        verbose=1
    )
]
history_class = model_class.fit(
    X_train_c, y_train_c, 
    epochs=500, 
    batch_size=64,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# %%
# Évaluation Classification
loss_c, acc_c = model_class.evaluate(X_test_c, y_test_c)
print(f"\nAccuracy sur le Test Set : {acc_c:.4f}")

# Plot
plt.plot(history_class.history['loss'], label='train_loss')
plt.plot(history_class.history['val_loss'], label='val_loss')
plt.title('Perte Classification')
plt.legend()
plt.show()

# %% [markdown]
# ## 3. PARTIE 2 : RÉGRESSION
# ---
# 
# Objectif : Prédire les 4 variables continues simultanément.

# %%
# Split des données
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg.values, y_reg.values, test_size=0.15, random_state=42
)

# %%
model_reg = Sequential([
    Input(shape=(X_train_r.shape[1],)),
    
    Dense(256),
    BatchNormalization(),
    Activation('swish'),
    Dropout(0.3),
    
    Dense(128),
    BatchNormalization(),
    Activation('swish'),
    Dropout(0.2),
    
    Dense(64),
    BatchNormalization(),
    Activation('swish'),
    Dropout(0.1),
    
    Dense(32, activation='swish'),
    Dense(4) # Sortie pour les 4 cibles
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model_reg.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
model_reg.summary()

# %%
# Entraînement
history_reg = model_reg.fit(
    X_train_r, y_train_r, 
    epochs=50, 
    batch_size=32, 
    validation_split=0.1,
    callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=30,
        min_lr=1e-6,
        verbose=1
    )
    ],
    verbose=1
)

# %%
# Évaluation Régression
y_pred_r = model_reg.predict(X_test_r)

# Pas d'inverse_transform car les y ne sont pas normalisés
y_pred_orig = y_pred_r
y_test_orig = y_test_r

target_names = ['Quiz_Score_Avg', 'Project_Grade', 'Satisfaction_Rating', 'Time_Spent_Hours']
for i, name in enumerate(target_names):
    rmse = np.sqrt(mean_squared_error(y_test_orig[:, i], y_pred_orig[:, i]))
    r2 = r2_score(y_test_orig[:, i], y_pred_orig[:, i])
    print(f"{name} -> RMSE: {rmse:.4f}, R2: {r2:.4f}")

# Plot Loss Régression
plt.plot(history_reg.history['loss'], label='train_mse')
plt.plot(history_reg.history['val_loss'], label='val_mse')
plt.title('Loss Régression')
plt.legend()
plt.show()


