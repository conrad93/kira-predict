from flask import Blueprint, request, jsonify
from services.soccer import get_model, predict_v0, predict_v1, predict_v2

predict_bp = Blueprint('predict', __name__)

@predict_bp.route('/predict/<version>', methods=['POST'])
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