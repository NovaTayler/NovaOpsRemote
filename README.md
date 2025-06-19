# NovaOpsRemote

This repo includes the full OmniMesh backend powered by FastAPI and two UIs:
- a Flask web dashboard in `novadash`
- a simple Tkinter desktop client (`NovaShellUI.app`)

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

