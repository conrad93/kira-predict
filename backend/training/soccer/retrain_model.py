import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

matches_path = 'data/soccer/Matches.csv'
future_path = 'data/soccer/future_results.csv'

df = pd.read_csv(matches_path, low_memory=False)
df = df.dropna(subset=['FTResult'])
df = df.drop(columns=['Division', 'MatchDate', 'MatchTime', 'HTResult'], errors='ignore')

if os.path.exists(future_path):
    future = pd.read_csv(future_path)
    future = future.rename(columns={
        'home_team': 'HomeTeam',
        'away_team': 'AwayTeam',
        'OddHome': 'OddHome',
        'OddDraw': 'OddDraw',
        'OddAway': 'OddAway',
        'actual': 'FTResult'
    })
    df = pd.concat([df, future], ignore_index=True)

df = pd.get_dummies(df, columns=['HomeTeam', 'AwayTeam'])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['FTResult'])

X = df.drop(columns=['FTResult'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = xgb.XGBClassifier(eval_metric='mlogloss')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Retrained model — Accuracy: {accuracy:.2f}")

# Ensure models folder exists
os.makedirs('models/soccer', exist_ok=True)

joblib.dump(model, 'models/soccer/model_v2.joblib')
joblib.dump(label_encoder, 'models/soccer/label_encoder.joblib')
print("💾 Updated model and label encoder saved.")