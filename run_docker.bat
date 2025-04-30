@echo off
echo Building Docker image...
docker-compose build

echo Starting Docker container...
docker-compose up

echo To stop the container, press Ctrl+C and then run: docker-compose down 