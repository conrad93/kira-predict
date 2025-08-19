from flask import Blueprint, jsonify
import subprocess

retrain_bp = Blueprint('retrain', __name__)

@retrain_bp.route('/api/retrain', methods=['POST'])
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