FROM python:3.11-slim
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Launch the OmniMesh backend (which also mounts the NovaDash UI)
CMD ["python", "-m", "omnimesh.backend"]
