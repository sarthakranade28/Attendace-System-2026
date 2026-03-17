# Set TensorFlow environment variables BEFORE any TF/Keras imports
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import sqlite3
import datetime
import pickle
import logging
from pathlib import Path
from mtcnn import MTCNN
from keras_facenet import FaceNet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE = 'attendance.db'

def check_models_exist():
    """Check if trained models exist"""
    models_path = Path("models")
    required_files = ["svm.pkl", "encoder.pkl"]
    
    for file in required_files:
        if not (models_path / file).exists():
            logger.error(f"Model file not found: {file}")
            return False
    return True

def init_db():
    """Initialize database for attendance"""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        return False

def take_attendance():
    """Main attendance capture function"""
    
    # Verify models exist
    if not check_models_exist():
        print("Error: Trained models not found. Please train the model first.")
        return False
    
    # Initialize database
    if not init_db():
        print("Error: Could not initialize database")
        return False
    
    try:
        # Load models
        print("Loading trained models...")
        svm = pickle.load(open("models/svm.pkl", "rb"))
        encoder = pickle.load(open("models/encoder.pkl", "rb"))
        
        # Initialize detection and embedding models
        detector = MTCNN()
        embedder = FaceNet()
        
        logger.info("Starting attendance capture")
        print("\n✓ Models loaded successfully")
        print("Starting attendance capture...")
        print("Press ESC to stop\n")
        
        # Open camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            logger.error("Could not open camera")
            print("Error: Could not access camera")
            return False
        
        present = set()  # Track who's already marked present
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                logger.error("Failed to read frame from camera")
                break
            
            frame_count += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb)
            
            # Process detected faces
            for face in faces:
                x, y, w, h = face['box']
                conf = face['confidence']
                
                # Only process confident detections
                if conf < 0.9:
                    continue
                
                face_img = rgb[y:y+h, x:x+w]
                
                if face_img.size == 0:
                    continue
                
                face_img = cv2.resize(face_img, (160, 160))
                
                try:
                    embedding = embedder.embeddings([face_img])[0]
                    embedding = embedding.reshape(1, -1)
                    
                    # Predict
                    pred = svm.predict(embedding)
                    prob = svm.predict_proba(embedding).max()
                    
                    # Mark as present if confidence is high
                    if prob > 0.7:
                        name = encoder.inverse_transform(pred)[0]
                        
                        if name not in present:
                            present.add(name)
                            cursor.execute(
                                'INSERT INTO attendance (name, timestamp) VALUES (?, ?)',
                                (name, datetime.datetime.now())
                            )
                            conn.commit()
                            logger.info(f"Attendance marked for {name}")
                            print(f"✓ {name} marked present")
                        
                        # Display name in green
                        cv2.putText(frame, name, (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        # Confidence too low, show as unknown
                        cv2.putText(frame, "Unknown", (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
                except Exception as e:
                    logger.error(f"Error processing face: {str(e)}")
                    continue
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Display frame with statistics
            text = f"Attendance: {len(present)} marked | Press ESC to stop"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
            
            cv2.imshow("Attendance System", frame)
            
            # Press ESC to exit
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        conn.close()
        
        print(f"\n✓ Attendance capture completed")
        print(f"  - Total people marked present: {len(present)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Attendance process failed: {str(e)}")
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = take_attendance()
    exit(0 if success else 1)

cap.release()
cv2.destroyAllWindows()