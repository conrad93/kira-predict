from flask import Flask, jsonify, request
from flask_cors import CORS
from joblib import load
import pandas as pd
from datetime import datetime
import csv, os, subprocess

app = Flask(__name__)
CORS(app)

static_model = None
real_model = None
model_v2 = None

label_encoder = load('label_encoder.joblib')

def get_model(version):
    global static_model, real_model, model_v2
    if version == "0":
        static_model = static_model or load('model.joblib')
        return static_model
    elif version == "1":
        real_model = real_model or load('model_real.joblib')
        return real_model
    elif version == "2":
        model_v2 = model_v2 or load('model_v2.joblib')
        return model_v2
    else:
        return None

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

@app.route('/api/predict/<version>', methods=['POST'])
def predict(version):
    data = request.json
    model = get_model(version)
    if not model:
        return jsonify({"error": "Invalid version"}), 400

    if version == "0":
        return predict_v0(model, data)
    elif version == "1":
        return predict_v1(model, data)
    elif version == "2":
        return predict_v2(model, data)
    else:
        return jsonify({"message": "Invalid version."})
    
@app.route('/api/log', methods=['POST'])
def log_result():
    try:
        data = request.json

        home_team = data['home_team']
        away_team = data['away_team']
        OddHome = float(data['OddHome'])
        OddDraw = float(data['OddDraw'])
        OddAway = float(data['OddAway'])
        predicted_result = data['predicted']
        actual_result = data['actual']
        bet_time = data.get('bet_time', datetime.now().isoformat())

        timestamp = datetime.now().isoformat()

        log_row = {
            "timestamp": timestamp,
            "bet_time": bet_time,
            "home_team": home_team,
            "away_team": away_team,
            "OddHome": OddHome,
            "OddDraw": OddDraw,
            "OddAway": OddAway,
            "predicted": predicted_result,
            "actual": actual_result
        }

        log_file = 'data/future_results.csv'
        file_exists = os.path.isfile(log_file)

        with open(log_file, mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=log_row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_row)
        
        return jsonify({"message": "Result logged successfully", "data": log_row})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/check_team', methods=['GET'])
def check_team():
    query = request.args.get('team')
    if not query:
        return jsonify({"error": "Missing 'team' query parameter"}), 400
    
    words = query.lower().split()

    def fuzzy_match(value):
        value_lower = str(value).lower()
        return all(word in value_lower for word in words)
    
    matches_result = []
    future_result = []

    try:
        df_matches = pd.read_csv('data/Matches.csv', low_memory=False)
        matched_home = df_matches['HomeTeam'].dropna().apply(fuzzy_match)
        matched_away = df_matches['AwayTeam'].dropna().apply(fuzzy_match)
        matched_teams = pd.concat([
            df_matches.loc[matched_home, 'HomeTeam'],
            df_matches.loc[matched_away, 'AwayTeam']
        ])
        matches_result = matched_teams.unique().tolist()
    except Exception as e:
        print(f"Error reading Matches.csv: {e}")

    try:
        df_future = pd.read_csv('data/future_results.csv')
        matched_home = df_future['home_team'].dropna().apply(fuzzy_match)
        matched_away = df_future['away_team'].dropna().apply(fuzzy_match)
        matched_teams = pd.concat([
            df_future.loc[matched_home, 'home_team'],
            df_future.loc[matched_away, 'away_team']
        ])
        future_result = matched_teams.unique().tolist()
    except Exception as e:
        print(f"Error reading future_results.csv: {e}")

    all_matches = list(set(matches_result + future_result))

    return jsonify({
        "query": query,
        "matches_in_matches_csv": matches_result,
        "matches_in_future_csv": future_result,
        "all_matches": all_matches,
        "exists": len(all_matches) > 0
    })

@app.route('/api/retrain', methods=['POST'])
def retrain_model():
    try:
        result = subprocess.run(
            ['python', 'retrain_model.py'],
            capture_output=True,
            text=True
        )
        return jsonify({
            "message": "Retraining triggered.",
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Flask Server...")
    app.run(debug=True)