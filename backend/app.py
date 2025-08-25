from flask import Flask
from flask_cors import CORS

from routes import soccer_bp

app = Flask(__name__)
CORS(app)

app.register_blueprint(soccer_bp)

if __name__ == '__main__':
    print("🚀 Starting Flask Server...")
    app.run(debug=True)