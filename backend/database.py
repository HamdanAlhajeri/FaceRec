import sqlite3
import numpy as np
import json
import os

class FaceDatabase:
    def __init__(self, db_path='faces.db'):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create faces table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            embedding BLOB,
            image_path TEXT,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        conn.close()

    def add_face(self, name, embedding, image_path, confidence):
        """Add a new face to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert numpy array to bytes
        embedding_bytes = embedding.tobytes()
        
        cursor.execute('''
        INSERT INTO faces (name, embedding, image_path, confidence)
        VALUES (?, ?, ?, ?)
        ''', (name, embedding_bytes, image_path, confidence))
        
        conn.commit()
        conn.close()

    def get_all_faces(self):
        """Retrieve all faces from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM faces')
        faces = cursor.fetchall()
        
        # Convert bytes back to numpy array
        processed_faces = []
        for face in faces:
            embedding = np.frombuffer(face[2], dtype=np.float32)
            processed_faces.append({
                'id': face[0],
                'name': face[1],
                'embedding': embedding,
                'image_path': face[3],
                'confidence': face[4],
                'created_at': face[5]
            })
        
        conn.close()
        return processed_faces

    def search_similar_faces(self, query_embedding, threshold=0.6):
        """Search for similar faces in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM faces')
        faces = cursor.fetchall()
        
        similar_faces = []
        for face in faces:
            embedding = np.frombuffer(face[2], dtype=np.float32)
            similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
            
            if similarity >= threshold:
                similar_faces.append({
                    'id': face[0],
                    'name': face[1],
                    'similarity': float(similarity),
                    'image_path': face[3],
                    'confidence': face[4]
                })
        
        conn.close()
        return sorted(similar_faces, key=lambda x: x['similarity'], reverse=True)

    def delete_face(self, face_id):
        """Delete a face from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM faces WHERE id = ?', (face_id,))
        
        conn.commit()
        conn.close()

    def update_face(self, face_id, name=None, embedding=None, image_path=None, confidence=None):
        """Update face information in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        updates = []
        values = []
        
        if name is not None:
            updates.append('name = ?')
            values.append(name)
        if embedding is not None:
            updates.append('embedding = ?')
            values.append(embedding.tobytes())
        if image_path is not None:
            updates.append('image_path = ?')
            values.append(image_path)
        if confidence is not None:
            updates.append('confidence = ?')
            values.append(confidence)
            
        if updates:
            query = f"UPDATE faces SET {', '.join(updates)} WHERE id = ?"
            values.append(face_id)
            cursor.execute(query, values)
            
        conn.commit()
        conn.close() 