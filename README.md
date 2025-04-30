# Face Recognition System

A web application for face detection and recognition using deep learning models. This system can detect faces in images, extract facial features, and compare facial similarities against reference images.

## Features

- Real-time face detection in uploaded images
- Facial feature extraction using ArcFace model
- Similarity comparison between detected faces and reference faces
- Database storage for detected faces
- Interactive web interface

## Docker Setup (Recommended)

The easiest way to run this application is with Docker:

### Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop/)
- [Docker Compose](https://docs.docker.com/compose/install/) (included with Docker Desktop)

### Building and Running

1. Clone this repository
2. Ensure you have sample face images in the `backend/100060861` and the other faces directory (the system needs reference faces)
3. Run the application using Docker:

**Windows:**
```
docker-compose build
docker-compose up -d
```

**Linux/Mac:**
```
docker-compose build
docker-compose up -d
```

4. Access the application at [http://localhost](http://localhost)

### Troubleshooting Common Issues

- If you encounter 500 errors when processing images, ensure:
  - You have face reference images in at least one of the reference folders (`100060861`, `100063264`, or `100063275`)
  - The server container has network access to download model files (or they're pre-downloaded)
  - Check logs with `docker logs facerec-server-1`

- If the website doesn't appear, check:
  - All containers are running with `docker ps`
  - Nginx is correctly serving files with `docker exec facerec-nginx-1 ls -la /usr/share/nginx/html`

### Stopping the Application

```
docker-compose down
```

## Manual Setup

If you prefer to run without Docker:

### Backend Setup

1. Navigate to the backend directory:
```
cd backend
```

2. Create a virtual environment:
```
python -m venv venv
```

3. Activate the virtual environment:
```
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

4. Install dependencies:
```
pip install -r requirements.txt
```

5. Ensure you have sample face images in the `100060861` directory

6. Run the Flask server:
```
python server.py
```

The API will be available at [http://localhost:5000](http://localhost:5000)

### Frontend Setup

1. Navigate to the client directory:
```
cd client
```

2. Install dependencies:
```
npm install
```

3. Start the development server:
```
npm start
```

4. Build the production files (for nginx to serve):
```
npm run build
```

The frontend development server will be available at [http://localhost:3000](http://localhost:3000)

## Architecture

The application consists of three main services:

1. **Backend (Python/Flask)**: Handles face detection, feature extraction, and similarity comparison using deep learning models
2. **Frontend (React)**: Provides the user interface for uploading images and viewing results
3. **Nginx**: Serves the static frontend files and routes API requests to the backend

## API Endpoints

- `/api/process_image` - Process an image for face detection and analysis
- `/faces` - Get all faces from the database
- `/faces/<face_id>` - Delete or update a specific face record
- `/reference-folders` - Get available reference image folders

## Technologies Used

- **Frontend**: React, Material-UI
- **Backend**: Flask, TensorFlow, OpenCV, DeepFace, RetinaFace
- **Database**: SQLite
- **Containerization**: Docker, Docker Compose
- **Web Server**: Nginx

## License

This project is open source and available under the MIT License.
