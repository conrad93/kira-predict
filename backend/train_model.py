import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from joblib import dump

data = pd.DataFrame({
    'team_goals': [2, 1, 3, 0, 4, 2, 1],
    'opponent_goals': [1, 2, 0, 3, 1, 0, 2],
    'win': [1, 0, 1, 0, 1, 1, 0]  
})

X = data[['team_goals', 'opponent_goals']]
y = data['win']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

dump(model, 'model.joblib')

print("✅ Model trained and saved as model.joblib")