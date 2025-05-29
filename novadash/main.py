from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from subprocess import run, PIPE
from pathlib import Path
from datetime import datetime

app = FastAPI()
templates = Jinja2Templates(directory="novadash/templates")
log_path = Path("novadash/command_log.txt")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "message": ""})

@app.post("/send", response_class=HTMLResponse)
async def send_command(request: Request, command: str = Form(...)):
    now = datetime.utcnow().isoformat()
    try:
        result = run(command, shell=True, stdout=PIPE, stderr=PIPE, text=True, timeout=10)
        output = result.stdout.strip() or result.stderr.strip()
        log_entry = f"[{now}] CMD: {command}\nOUT: {output}\n\n"
        log_path.write_text(log_path.read_text() + log_entry if log_path.exists() else log_entry)
        message = f"<pre>{output}</pre>"
    except Exception as e:
        message = f"❌ Error: {e}"
    return templates.TemplateResponse("index.html", {"request": request, "message": message})
