import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

data = pd.read_csv('data/SP1.csv')
print("✅ Loaded data")

data = data[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'B365H', 'B365D', 'B365A']].dropna()

X = data[['B365H', 'B365D', 'B365A']]
y = data['FTR'].map({'H': 1, 'D': 0, 'A': -1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

dump(model, 'model_real.joblib')

print("✅ Model trained and saved as model_real.joblib")