import os
import tensorflow as tf
import logging
import traceback

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
tf.get_logger().setLevel(logging.ERROR)

# Set environment variable to use pre-downloaded models
os.environ["RETINAFACE_WEIGHT_PATH"] = "/root/.deepface/weights/retinaface.h5"
os.environ["DEEPFACE_WEIGHTS"] = "/root/.deepface/weights"

from flask import Flask, request, jsonify
from flask_cors import CORS
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

# Available reference folders
REFERENCE_FOLDERS = ['100060861', '100063275', '100063264']

app = Flask(__name__)
# Configure CORS to allow requests from React frontend
CORS(app, resources={r"/*": {"origins": "*"}})
db = FaceDatabase()

@app.before_request
def print_request_info():
    print('Headers:', request.headers)
    print('Body:', request.get_data())

def compute_similarity(embedding1, embedding2):
    try:
        return np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
    except Exception as e:
        print(f"Error computing similarity: {str(e)}")
        return 0.0

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

def get_random_reference_face(folder=None):
    """Get a random face from the input directory for comparison"""
    try:
        # Use the specified folder or find one with images
        if folder and folder in REFERENCE_FOLDERS:
            input_dir = folder
        else:
            # Find a folder that has PNG files
            folders_with_images = []
            for ref_folder in REFERENCE_FOLDERS:
                if os.path.exists(ref_folder):
                    files = [f for f in os.listdir(ref_folder) if f.endswith('.png')]
                    if files:
                        folders_with_images.append(ref_folder)
            
            if not folders_with_images:
                print("No reference folders with images found")
                return None
            
            input_dir = folders_with_images[0]
        
        print(f"Using reference folder: {input_dir}")
        
        if not os.path.exists(input_dir):
            print(f"Reference folder does not exist: {input_dir}")
            return None

        # Get all face images
        face_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
        if not face_files:
            print(f"No face files found in folder: {input_dir}")
            return None

        # Select a random face
        random_face = random.choice(face_files)
        face_path = os.path.join(input_dir, random_face)
        print(f"Selected reference face: {face_path}")

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
                'image_path': face_path,
                'folder': input_dir
            }
        except Exception as e:
            print(f"Error processing reference face: {str(e)}")
            print(traceback.format_exc())
            return None
    except Exception as e:
        print(f"Error in get_random_reference_face: {str(e)}")
        print(traceback.format_exc())
        return None

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        print("Starting process_image function")
        # Get the image from the request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
            
        # Get reference folder if provided
        reference_folder = data.get('reference_folder')
        print(f"Processing image with reference folder: {reference_folder}")

        try:
            # Convert base64 image to numpy array
            image_data = data['image'].split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            image_np = np.array(image)
        except Exception as e:
            print(f"Error processing input image: {str(e)}")
            print(traceback.format_exc())
            return jsonify({'error': 'Invalid image format'}), 400

        # Save the image temporarily
        temp_path = 'temp_query.jpg'
        try:
            cv2.imwrite(temp_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            print(f"Saved temporary image to: {temp_path}")
        except Exception as e:
            print(f"Error saving temporary image: {str(e)}")
            print(traceback.format_exc())
            return jsonify({'error': 'Error saving temporary image'}), 500

        try:
            # Step 1: Extract all faces from the query image
            query_faces = RetinaFace.extract_faces(temp_path)
            print(f"Number of faces detected: {len(query_faces)}")
            
            if len(query_faces) == 0:
                return jsonify({'error': 'No faces detected in the query image'}), 400
        except Exception as e:
            print(f"Error extracting faces: {str(e)}")
            print(traceback.format_exc())
            return jsonify({'error': 'Error extracting faces from image'}), 500

        # Step 2: Get a random reference face for comparison
        reference_face = get_random_reference_face(reference_folder)
        if not reference_face:
            return jsonify({'error': 'Could not get reference face'}), 500

        # Step 3: Process each detected face
        face_results = []
        face_image_paths = []
        
        for i, face in enumerate(query_faces):
            try:
                print(f"Processing face {i+1}")
                # Save each detected face as an image
                face_filename = f"detected_face_{uuid.uuid4()}.png"
                face_path = os.path.join('input', face_filename)
                
                # Convert face array to BGR for saving
                try:
                    face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(face_path, face_bgr)
                    face_image_paths.append(face_path)
                    print(f"Saved face {i+1} to {face_path}")
                except Exception as e:
                    print(f"Error saving face {i+1}: {str(e)}")
                    print(traceback.format_exc())
                    continue

                # Get face embedding
                try:
                    print(f"Getting embedding for face {i+1}")
                    face_signature = DeepFace.represent(
                        face,
                        model_name='ArcFace',
                        enforce_detection=False
                    )
                    print(f"Got embedding for face {i+1}")
                    
                    face_embedding = np.array(face_signature[0]['embedding'])
                    face_confidence = face_signature[0]['face_confidence']
                except Exception as e:
                    print(f"Error getting embedding for face {i+1}: {str(e)}")
                    print(traceback.format_exc())
                    continue

                # Compute similarity
                try:
                    similarity = compute_similarity(face_embedding, reference_face['embedding'])
                    print(f"Computed similarity for face {i+1}: {similarity}")
                except Exception as e:
                    print(f"Error computing similarity for face {i+1}: {str(e)}")
                    print(traceback.format_exc())
                    continue

                face_results.append({
                    'face_id': i + 1,
                    'confidence': float(face_confidence),
                    'similarity': float(similarity),
                    'image_path': face_path
                })
                print(f"Successfully processed face {i+1}")
            except Exception as e:
                print(f"Error processing face {i+1}: {str(e)}")
                print(traceback.format_exc())
                continue

        if not face_results:
            return jsonify({'error': 'Failed to process any faces'}), 500

        try:
            print("Preparing response")
            # Step 4: Find the face with highest confidence
            best_face = max(face_results, key=lambda x: x['confidence'])
            best_similarity_face = max(face_results, key=lambda x: x['similarity'])

            # Read and encode images as base64
            with open(best_face['image_path'], 'rb') as img_file:
                best_face_base64 = base64.b64encode(img_file.read()).decode('utf-8')

            with open(best_similarity_face['image_path'], 'rb') as img_file:
                best_similarity_base64 = base64.b64encode(img_file.read()).decode('utf-8')

            with open(reference_face['image_path'], 'rb') as ref_img_file:
                reference_face_base64 = base64.b64encode(ref_img_file.read()).decode('utf-8')

            print("Successfully prepared response")
        except Exception as e:
            print(f"Error preparing response: {str(e)}")
            print(traceback.format_exc())
            return jsonify({'error': 'Error preparing response'}), 500

        # Clean up
        try:
            os.remove(temp_path)
            print("Cleaned up temporary files")
        except Exception as e:
            print(f"Error cleaning up temporary file: {str(e)}")

        print("Returning successful response")
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
            'reference_folder': reference_face['folder']
        })

    except Exception as e:
        print(f"Error in process_image: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/process_image', methods=['POST'])
def api_process_image():
    try:
        print("Received request at /api/process_image")
        print("Request JSON:", request.get_json())
        
        # Detailed debugging for reference folders
        print("Checking reference folders for images:")
        for folder in REFERENCE_FOLDERS:
            if os.path.exists(folder):
                files = [f for f in os.listdir(folder) if f.endswith('.png')]
                print(f"Folder {folder}: {len(files)} PNG files found")
                if len(files) > 0:
                    print(f"Example files: {files[:3]}")
            else:
                print(f"Folder {folder} does not exist")
        
        # Call the process_image function
        try:
            response = process_image()
            print("Process image response type:", type(response))
            return response
        except Exception as e:
            print(f"Error in process_image function: {str(e)}")
            print(traceback.format_exc())
            return jsonify({'error': f"Error in process_image: {str(e)}"}), 500
            
    except Exception as e:
        error_msg = f"API Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({'error': error_msg}), 500

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

@app.route('/reference-folders', methods=['GET'])
def get_reference_folders():
    """Get a list of available reference folders"""
    try:
        # Return only folders that exist
        existing_folders = [folder for folder in REFERENCE_FOLDERS if os.path.exists(folder)]
        return jsonify({
            'folders': existing_folders
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return "Face Recognition API"

if __name__ == '__main__':
    # Create input directory if it doesn't exist
    os.makedirs('input', exist_ok=True)
    
    # Check reference folders
    print("Checking reference folders...")
    for folder in REFERENCE_FOLDERS:
        if not os.path.exists(folder):
            print(f"Warning: Reference folder {folder} does not exist")
            os.makedirs(folder, exist_ok=True)
        else:
            files = [f for f in os.listdir(folder) if f.endswith('.png')]
            print(f"Found {len(files)} PNG files in {folder}")
    
    # Initialize DeepFace model
    print("Initializing DeepFace model...")
    try:
        # Warm up the model with a test call
        test_img = np.zeros((224, 224, 3), dtype=np.uint8)
        DeepFace.represent(test_img, model_name='ArcFace', enforce_detection=False)
        print("DeepFace model initialized successfully")
    except Exception as e:
        print(f"Warning: Failed to initialize DeepFace model: {str(e)}")
        print(traceback.format_exc())
    
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)