from flask import Flask
from flask_cors import CORS

from routes.predict import predict_bp
from routes.log import log_bp
from routes.check_team import check_team_bp
from routes.retrain import retrain_bp

app = Flask(__name__)
CORS(app)

app.register_blueprint(predict_bp)
app.register_blueprint(log_bp)
app.register_blueprint(check_team_bp)
app.register_blueprint(retrain_bp)

if __name__ == '__main__':
    print("🚀 Starting Flask Server...")
    app.run(debug=True)