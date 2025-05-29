from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI()

# Mount the templates directory
templates = Jinja2Templates(directory="novadash/templates")

# Serve static files if needed (optional)
if os.path.exists("novadash/static"):
    app.mount("/static", StaticFiles(directory="novadash/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "message": "Welcome to NovaShell 🔥"})

@app.post("/send")
async def send_command(request: Request, command: str = Form(...)):
    os.system(command)  # Optional: log/validate commands instead
    return templates.TemplateResponse("index.html", {"request": request, "message": f"Executed: {command}"})
