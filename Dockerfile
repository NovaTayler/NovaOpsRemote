FROM python:3.11-slim
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Launch the OmniMesh backend which exposes the API and dashboard
CMD ["uvicorn", "omnimesh.backend:app", "--host", "0.0.0.0", "--port", "8080"]
