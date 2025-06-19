from flask import Flask
import os

app = Flask(__name__)

@app.route('/')
def index():
    return "🚀 Humanitas Cloud Run is LIVE!"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
