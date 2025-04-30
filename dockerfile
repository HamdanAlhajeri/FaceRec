# Stage 1: Build the React frontend
FROM node:16-alpine as frontend-build

WORKDIR /app/frontend

# Copy frontend files
COPY client/package*.json ./
RUN npm ci

COPY client/ ./
RUN npm run build

# Stage 2: Set up the Python backend with the frontend
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Set up Nginx to serve frontend
COPY --from=frontend-build /app/frontend/build /var/www/html
COPY nginx.conf /etc/nginx/sites-available/default

# Copy backend requirements
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ ./

# Create required directories
RUN mkdir -p input
RUN mkdir -p 100060861

# Create startup script
RUN echo '#!/bin/bash\n\
service nginx start\n\
python server.py\n\
' > /app/start.sh && chmod +x /app/start.sh

# Expose ports for backend and frontend
EXPOSE 5000 80

# Start both services
CMD ["/app/start.sh"]