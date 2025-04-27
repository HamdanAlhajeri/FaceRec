import requests
import base64
import json
import os
import cv2
import numpy as np
from PIL import Image

def send_image_to_server(image_path):
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} does not exist")
        return

    print(f"Sending image: {image_path}")
    
    # Read the image file
    with open(image_path, 'rb') as image_file:
        # Convert image to base64
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Prepare the request data
        data = {
            'image': f'data:image/png;base64,{encoded_string}'
        }
        
        # Send POST request to the server
        response = requests.post('http://localhost:5000/process_image', json=data)
        
        # Print the response
        print('Status Code:', response.status_code)
        print('Response:', response.json())

if __name__ == '__main__':
    # Use the actual image file
    send_image_to_server('frame_0000.png') 