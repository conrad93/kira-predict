from flask import Blueprint, request, jsonify
from datetime import datetime
import csv, os

log_bp = Blueprint('log', __name__)

@log_bp.route('/log', methods=['POST'])
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

        log_file = 'data/soccer/future_results.csv'
        file_exists = os.path.isfile(log_file)

        with open(log_file, mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=log_row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_row)
        
        return jsonify({"message": "Result logged successfully", "data": log_row})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500