# NovaOpsRemote

This repo includes the full OmniMesh backend powered by FastAPI and two UIs:
- a Flask web dashboard in `novadash`
- a simple Tkinter desktop client (`NovaShellUI.app`)
The Flask dashboard is mounted at `/dash` when the backend starts.

## Running OmniMesh backend

```bash
python -m omnimesh.backend
```

## Running NovaDash web UI

```bash
python -m novadash.main
```

## Building the desktop `.app`

To build a macOS `.app` for the Tkinter UI, install `py2app` and run:

```bash
python setup.py py2app
```

The built application will appear in the `dist` directory.

## Docker Compose

You can start both the backend and the NovaDash UI using docker-compose:

```bash
docker-compose up --build
```

Or simply run the helper script:

```bash
./deploy.sh
```


Alternatively, run the services directly (without Docker):

```bash
./run.sh
```

## Deploying to Cloud Run

Build the container image and deploy it using the Google Cloud CLI:

```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/novaops
gcloud run deploy novaops --image gcr.io/PROJECT_ID/novaops --platform managed --region us-central1 --allow-unauthenticated
```

The service will expose the OmniMesh API at `/` and the NovaDash dashboard at `/dash`.

