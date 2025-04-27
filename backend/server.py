from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import base64
from io import BytesIO
from PIL import Image
import uuid
import random

from deepface import DeepFace
from retinaface import RetinaFace
from database import FaceDatabase

app = Flask(__name__)
CORS(app)
db = FaceDatabase()

def compute_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))

def extract_imgs(img_path):
    # Lower the detection threshold to detect more faces (default is 0.9)
    faces = RetinaFace.extract_faces(
        img_path=img_path,
        threshold=0.8
    )
    n = len(faces)

    plt.figure(figsize=(20, 20))  # Make the figure larger
    for i, f in enumerate(faces):
        plt.subplot((n+4)//5, 5, i + 1)  # Create a grid that can fit more faces
        plt.imshow(f)
        plt.axis('off')
        plt.title(f"Face: {i + 1}")
    plt.tight_layout()
    plt.show()

    return faces

def get_random_reference_face():
    """Get a random face from the input directory for comparison"""
    input_dir = '100060861'
    if not os.path.exists(input_dir):
        return None

    # Get all face images
    face_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    if not face_files:
        return None

    # Select a random face
    random_face = random.choice(face_files)
    face_path = os.path.join(input_dir, random_face)

    # Generate embedding for the random face
    try:
        face_signature = DeepFace.represent(
            img_path=face_path,
            model_name='ArcFace',
            enforce_detection=False
        )
        return {
            'embedding': np.array(face_signature[0]['embedding']),
            'confidence': face_signature[0]['face_confidence'],
            'image_path': face_path
        }
    except Exception as e:
        print(f"Error processing reference face: {e}")
        return None

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # Get the image from the request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        # Convert base64 image to numpy array
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        image_np = np.array(image)

        # Save the image temporarily
        temp_path = 'temp_query.jpg'
        cv2.imwrite(temp_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        print(f"Saved temporary image to: {temp_path}")

        # Step 1: Extract all faces from the query image
        query_faces = RetinaFace.extract_faces(temp_path)
        print(f"Number of faces detected: {len(query_faces)}")
        
        if len(query_faces) == 0:
            return jsonify({'error': 'No faces detected in the query image'}), 400

        # Step 2: Get a random reference face for comparison
        reference_face = get_random_reference_face()
        if not reference_face:
            return jsonify({'error': 'Could not get reference face'}), 500

        # Step 3: Process each detected face
        face_results = []
        face_image_paths = []
        for i, face in enumerate(query_faces):
            # Save each detected face as an image
            face_filename = f"detected_face_{uuid.uuid4()}.png"
            face_path = os.path.join('input', face_filename)
            cv2.imwrite(face_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
            face_image_paths.append(face_path)
            face_signature = DeepFace.represent(
                face,
                model_name='ArcFace',
                enforce_detection=False
            )
            face_embedding = np.array(face_signature[0]['embedding'])
            face_confidence = face_signature[0]['face_confidence']

            # Compute similarity with reference face
            similarity = compute_similarity(face_embedding, reference_face['embedding'])

            face_results.append({
                'face_id': i + 1,
                'confidence': float(face_confidence),
                'similarity': float(similarity),
                'image_path': face_path
            })

        # Step 4: Find the face with highest confidence
        best_face = max(face_results, key=lambda x: x['confidence'])

        # Read the best face image and encode as base64
        with open(best_face['image_path'], 'rb') as img_file:
            best_face_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        # Find the face with the highest similarity
        best_similarity_face = max(face_results, key=lambda x: x['similarity'])
        with open(best_similarity_face['image_path'], 'rb') as img_file:
            best_similarity_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        # Read the reference face image and encode as base64
        with open(reference_face['image_path'], 'rb') as ref_img_file:
            reference_face_base64 = base64.b64encode(ref_img_file.read()).decode('utf-8')

        # Clean up
        os.remove(temp_path)

        return jsonify({
            'all_faces': face_results,
            'best_match': {
                'face_id': best_face['face_id'],
                'confidence': best_face['confidence'],
                'similarity': best_face['similarity'],
                'image_path': best_face['image_path'],
                'image_base64': f'data:image/png;base64,{best_face_base64}',
                'reference_face': reference_face['image_path']
            },
            'best_similarity_match': {
                'face_id': best_similarity_face['face_id'],
                'confidence': best_similarity_face['confidence'],
                'similarity': best_similarity_face['similarity'],
                'image_path': best_similarity_face['image_path'],
                'image_base64': f'data:image/png;base64,{best_similarity_base64}',
            },
            'reference_face_base64': f'data:image/png;base64,{reference_face_base64}',
        })

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/faces', methods=['GET'])
def get_faces():
    """Get all faces from the database"""
    try:
        faces = db.get_all_faces()
        return jsonify({
            'faces': [{
                'id': face['id'],
                'name': face['name'],
                'image_path': face['image_path'],
                'confidence': face['confidence'],
                'created_at': face['created_at']
            } for face in faces]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/faces/<int:face_id>', methods=['DELETE'])
def delete_face(face_id):
    """Delete a face from the database"""
    try:
        db.delete_face(face_id)
        return jsonify({'message': 'Face deleted successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/faces/<int:face_id>', methods=['PUT'])
def update_face(face_id):
    """Update face information in the database"""
    try:
        data = request.get_json()
        db.update_face(
            face_id,
            name=data.get('name'),
            image_path=data.get('image_path'),
            confidence=data.get('confidence')
        )
        return jsonify({'message': 'Face updated successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return "Face Recognition API"

if __name__ == '__main__':
    # Create input directory if it doesn't exist
    os.makedirs('input', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)