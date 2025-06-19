#!/bin/bash
set -e

# Deploy OmniMesh backend and NovaDash UI using Docker Compose

echo "Building containers..."
docker-compose build

echo "Starting services..."
docker-compose up -d

echo "Deployment complete!"
