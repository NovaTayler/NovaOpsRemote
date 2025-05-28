from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import subprocess
import json

app = FastAPI()
templates = Jinja2Templates(directory="templates")

with open("config.json", "r") as f:
    config = json.load(f)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/send", response_class=HTMLResponse)
async def send_command(request: Request, command: str = Form(...)):
    try:
        with open(config["command_file"], "w") as f:
            f.write(command + "\n")
        subprocess.run(["git", "add", config["command_file"]], cwd=config["repo_path"])
        subprocess.run(["git", "commit", "-m", "Command from dashboard"], cwd=config["repo_path"])
        subprocess.run(["git", "push", "origin", config["branch"]], cwd=config["repo_path"])
        message = "✅ Command sent successfully!"
    except Exception as e:
        message = f"❌ Error: {e}"
    return templates.TemplateResponse("index.html", {"request": request, "message": message})
