from .predict import predict_bp
from .log import log_bp
from .check_team import check_team_bp
from .retrain import retrain_bp

def register_routes(app):
    app.register_blueprint(predict_bp)
    app.register_blueprint(log_bp)
    app.register_blueprint(check_team_bp)
    app.register_blueprint(retrain_bp)