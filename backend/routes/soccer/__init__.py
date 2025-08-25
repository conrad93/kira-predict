from flask import Blueprint

from .predict import predict_bp
from .log import log_bp
from .check_team import check_team_bp
from .retrain import retrain_bp

soccer_bp = Blueprint('soccer_bp', __name__, url_prefix='/api/soccer')

# Register sub-blueprints
soccer_bp.register_blueprint(predict_bp)
soccer_bp.register_blueprint(log_bp)
soccer_bp.register_blueprint(check_team_bp)
soccer_bp.register_blueprint(retrain_bp)
