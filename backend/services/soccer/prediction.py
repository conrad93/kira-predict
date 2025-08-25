from flask import jsonify
import pandas as pd
from joblib import load

label_encoder = load('models/soccer/label_encoder.joblib')

def predict_v0(model, data):
    team_goals = data.get('team_goals', 0)
    opponent_goals = data.get('opponent_goals', 0)

    try:
        team_goals = int(team_goals)
        opponent_goals = int(opponent_goals)
    except ValueError:
        return jsonify({"error": "Invalid input values. Must be numbers."}), 400
    
    prediction = model.predict([[team_goals, opponent_goals]])

    result = "Win" if prediction[0] == 1 else "Loss"

    return jsonify({
        "team_goals": team_goals,
        "opponent_goals": opponent_goals,
        "prediction": result
    })

def predict_v1(model, data):
    try:
        home_odds = float(data.get('B365H'))
        draw_odds = float(data.get('B365D'))
        away_odds = float(data.get('B365A'))
    except (TypeError, ValueError):
        return jsonify({"error": "Missing or invalid odds"}), 400
    
    prediction = model.predict([[home_odds, draw_odds, away_odds]])
    result_map = {1: "Home Win", 0: "Draw", -1: "Away Win"}
    result = result_map.get(prediction[0], "Unknown")

    return jsonify({
        "input": {
            "B365H": home_odds,
            "B365D": draw_odds,
            "B365A": away_odds
        },
        "prediction": result
    })

def predict_v2(model, data):
    try:
        expected_columns = model.get_booster().feature_names

        home_team = data.get("home_team")
        away_team = data.get("away_team")
        home_odds = float(data.get("OddHome"))
        draw_odds = float(data.get("OddDraw"))
        away_odds = float(data.get("OddAway"))
    
        input_data = {col: 0 for col in expected_columns}

        input_data['OddHome'] = home_odds
        input_data['OddDraw'] = draw_odds
        input_data['OddAway'] = away_odds

        home_col = f'HomeTeam_{home_team}'
        away_col = f'AwayTeam_{away_team}'

        if home_col in input_data:
            input_data[home_col] = 1
        if away_col in input_data:
            input_data[away_col] = 1

        df_input = pd.DataFrame([input_data])
        prediction = model.predict(df_input)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        return jsonify({
            "input": data,
            "prediction": predicted_label
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500