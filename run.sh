#!/bin/bash
set -e

# Launch OmniMesh backend and NovaDash UI without Docker
python -m omnimesh.backend &
BACKEND_PID=$!
python -m novadash.main &
UI_PID=$!

trap 'kill $BACKEND_PID $UI_PID' EXIT
wait
