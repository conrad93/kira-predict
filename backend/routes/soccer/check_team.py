from flask import Blueprint, request, jsonify
import pandas as pd

check_team_bp = Blueprint('check_team', __name__)

@check_team_bp.route('/check_team', methods=['GET'])
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
        df_matches = pd.read_csv('data/soccer/Matches.csv', low_memory=False)
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
        df_future = pd.read_csv('data/soccer/future_results.csv')
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