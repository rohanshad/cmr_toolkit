#!/bin/bash
set -e

# Workstation guard — full test suite is only configured to run on jarvis_lambda
if [[ "$(hostname)" != *"jarvis"* ]]; then
    echo "Error: Docker test suite is configured to run on jarvis_lambda only."
    echo "Current host: $(hostname)"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "Generating .env from local_config.yaml..."
python tests/docker_prep.py

echo "Spinning up Docker test environment (Ubuntu 24.04)..."
docker compose --project-directory "$REPO_ROOT" -f tests/docker-compose.yml up --build --force-recreate --abort-on-container-exit

echo "Cleaning up..."
docker compose --project-directory "$REPO_ROOT" -f tests/docker-compose.yml down

echo "All preprocessing validation tests passed."
