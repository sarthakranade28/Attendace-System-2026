# Set TensorFlow environment variables BEFORE any TF/Keras imports
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import pickle
import logging
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from mtcnn import MTCNN
from keras_facenet import FaceNet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create models directory
Path("models").mkdir(exist_ok=True)

def train_model():
    """Train SVM model on captured face images"""
    
    dataset_path = Path("dataset")
    
    # Check if dataset exists
    if not dataset_path.exists() or not list(dataset_path.glob("*/")): 
        logger.error("No dataset found. Please capture face images first.")
        print("Error: No dataset found. Please capture face images first.")
        return False
    
    try:
        print("Initializing face detection and embedding models...")
        detector = MTCNN()
        embedder = FaceNet()
        
        X = []  # Feature embeddings
        y = []  # Labels (names)
        people_count = 0
        
        dataset_count = len(list(dataset_path.glob("*/")))
        
        # Check if there are at least 2 people for training
        if dataset_count < 2:
            logger.error(f"Need at least 2 people for training, but found only {dataset_count}")
            print(f"Error: You need at least 2 different people to train the model.")
            print(f"Currently have: {dataset_count} person(s)")
            print("Please add more people to the dataset and try again.")
            return False
        
        print(f"\nProcessing images from {dataset_count} people...")
        
        # Extract embeddings from all face images
        for person_dir in dataset_path.iterdir():
            if not person_dir.is_dir():
                continue
            
            person_name = person_dir.name
            image_files = list(person_dir.glob("*.jpg"))
            
            if not image_files:
                logger.warning(f"No images found for {person_name}")
                continue
            
            print(f"Processing {person_name} ({len(image_files)} images)...")
            people_count += 1
            
            for img_path in image_files:
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        logger.warning(f"Could not read image: {img_path}")
                        continue
                    
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    faces = detector.detect_faces(rgb)
                    
                    if faces:
                        # Get first face detection
                        x, y_coord, w, h = faces[0]['box']
                        face = rgb[y_coord:y_coord+h, x:x+w]
                        
                        # Resize to standard size
                        face = cv2.resize(face, (160, 160))
                        
                        # Get embedding
                        embedding = embedder.embeddings([face])[0]
                        
                        X.append(embedding)
                        y.append(person_name)
                        
                except Exception as e:
                    logger.error(f"Error processing {img_path}: {str(e)}")
                    continue
        
        if len(X) == 0:
            logger.error("No valid face embeddings extracted")
            print("Error: Could not extract face embeddings from images")
            return False
        
        print(f"\nExtracted {len(X)} face embeddings from {people_count} people")
        print("Training SVM classifier...")
        
        # Encode labels
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        
        # Train SVM model
        model = SVC(kernel='linear', probability=True, random_state=42)
        model.fit(X, y_encoded)
        
        # Save models
        models_path = Path("models")
        models_path.mkdir(exist_ok=True)
        
        pickle.dump(model, open(models_path / "svm.pkl", "wb"))
        pickle.dump(encoder, open(models_path / "encoder.pkl", "wb"))
        
        logger.info("Training completed successfully")
        print(f"\n✓ Model trained successfully!")
        print(f"  - Trained on {people_count} people")
        print(f"  - Used {len(X)} face samples")
        print(f"  - Models saved to 'models/' directory")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        print(f"Error during training: {str(e)}")
        return False

if __name__ == "__main__":
    success = train_model()
    exit(0 if success else 1)