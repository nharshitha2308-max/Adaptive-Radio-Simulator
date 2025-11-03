import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# --- 1. Load your dataset ---
df = pd.read_csv('modulation_dataset.csv')
print("Loaded data:", df.shape)
print(df.head())

# --- 2. Prepare features and labels (NO Bandwidth) ---
features = ['SNR_dB', 'Signal_Power_dBm', 'Channel_Type', 'Latency_ms']
X = df[features]
y = df['Modulation_Type']

# --- 3. One-Hot Encode Channel_Type ---
encoder = OneHotEncoder(sparse_output=False)
channel_encoded = encoder.fit_transform(X[['Channel_Type']])
channel_cols = encoder.get_feature_names_out(['Channel_Type'])

# Combine encoded channel columns with numeric ones
X_encoded = pd.concat(
    [X.drop('Channel_Type', axis=1).reset_index(drop=True),
     pd.DataFrame(channel_encoded, columns=channel_cols)],
    axis=1
)

# --- 4. Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# --- 5. Train Decision Tree Classifier ---
clf = DecisionTreeClassifier(max_depth=8, random_state=42)
clf.fit(X_train, y_train)

# --- 6. Evaluate ---
y_pred = clf.predict(X_test)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- 7. Save model and encoder (with absolute paths) ---
save_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(save_dir, 'modulation_model_new.joblib')
encoder_path = os.path.join(save_dir, 'channel_encoder.joblib')

joblib.dump(clf, model_path)
joblib.dump(encoder, encoder_path)

print("\n✅ Model and encoder saved successfully!")
print(f"📁 Model path: {model_path}")
print(f"📁 Encoder path: {encoder_path}")