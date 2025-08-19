import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the data
df = pd.read_csv('data/Matches.csv', low_memory=False)
df = df.dropna(subset=['FTResult'])
df = df.drop(columns=['Division', 'MatchDate', 'MatchTime', 'HTResult'])

# One-hot encode team names
df = pd.get_dummies(df, columns=['HomeTeam', 'AwayTeam'])

# Encode the match result
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['FTResult'])

# Features (with odds)
X = df.drop(columns=['FTResult'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train the model
model = xgb.XGBClassifier(eval_metric='mlogloss')
model.fit(X_train, y_train)

# Evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained — Accuracy: {accuracy:.2f}")

# Save model and encoder
joblib.dump(model, 'model_v2.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')
print("💾 Model and label encoder saved.")
