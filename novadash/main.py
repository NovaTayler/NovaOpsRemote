from dataclasses import dataclass
from flask import Flask, request, render_template
from subprocess import run, PIPE
from pathlib import Path
from datetime import datetime

@dataclass
class NovaDashConfig:
    log_path: Path = Path("novadash/command_log.txt")

config = NovaDashConfig()
app = Flask(__name__, template_folder="templates")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", message="")

@app.route("/send", methods=["POST"])
def send_command():
    command = request.form.get("command", "")
    now = datetime.utcnow().isoformat()
    try:
        result = run(command, shell=True, stdout=PIPE, stderr=PIPE, text=True, timeout=10)
        output = result.stdout.strip() or result.stderr.strip()
        log_entry = f"[{now}] CMD: {command}\nOUT: {output}\n\n"
        if config.log_path.exists():
            config.log_path.write_text(config.log_path.read_text() + log_entry)
        else:
            config.log_path.write_text(log_entry)
        message = f"<pre>{output}</pre>"
    except Exception as e:
        message = f"❌ Error: {e}"
    return render_template("index.html", message=message)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
