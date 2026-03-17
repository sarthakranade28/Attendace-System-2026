# Set TensorFlow environment variables BEFORE any TF/Keras imports
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, render_template, redirect, request, jsonify, flash, send_file
import sqlite3
import subprocess
import logging
import cv2
import pickle
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'attendance_system_secret_key'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DATASET_FOLDER'] = 'dataset'

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE = 'attendance.db'

# Create necessary directories
Path("uploads").mkdir(exist_ok=True)
Path("dataset").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)
Path("static/uploads").mkdir(parents=True, exist_ok=True)

# Global model instances for speed
_detector = None
_embedder = None

def get_detector():
    """Get or create MTCNN detector"""
    global _detector
    if _detector is None:
        logger.info("Loading MTCNN detector...")
        from mtcnn import MTCNN
        _detector = MTCNN()
    return _detector

def get_embedder():
    """Get or create FaceNet embedder"""
    global _embedder
    if _embedder is None:
        logger.info("Loading FaceNet embedder...")
        from keras_facenet import FaceNet
        _embedder = FaceNet()
    return _embedder

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_embedding(img_path):
    """Extract embedding from a single image"""
    try:
        detector = get_detector()
        embedder = get_embedder()
        
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)
        
        if not faces:
            return None
        
        x, y_coord, w, h = faces[0]['box']
        face = rgb[y_coord:y_coord+h, x:x+w]
        face = cv2.resize(face, (160, 160))
        embedding = embedder.embeddings([face])[0]
        
        return embedding
        
    except Exception as e:
        logger.error(f"Error extracting embedding: {str(e)}")
        return None

def get_attendance_records():
    """Retrieve all attendance records from database"""
    try:
        conn = sqlite3.connect(DATABASE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT name, timestamp FROM attendance ORDER BY timestamp DESC')
        records = cursor.fetchall()
        conn.close()
        return records
    except Exception as e:
        logger.error(f"Error retrieving attendance records: {str(e)}")
        return []

# Initialize database on startup
def init_db():
    """Initialize SQLite database with attendance table"""
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
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")

init_db()

def run_python_script(script_name):
    """Safely execute Python scripts using subprocess"""
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        logger.error(f"Script {script_name} timed out")
        return False, "", "Script execution timed out"
    except Exception as e:
        logger.error(f"Error running script {script_name}: {str(e)}")
        return False, "", str(e)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        files = request.files.getlist("images")
        
        if not name:
            flash("Please enter a name", "error")
            return redirect("/register")
        
        if not files or len(files) == 0:
            flash("Please upload at least one image", "error")
            return redirect("/register")
        
        # Create dataset directory for this person
        person_path = Path("dataset") / name
        person_path.mkdir(parents=True, exist_ok=True)
        
        saved_count = 0
        
        for file in files:
            if file and allowed_file(file.filename):
                try:
                    filename = secure_filename(f"{saved_count}.jpg")
                    filepath = person_path / filename
                    file.save(str(filepath))
                    saved_count += 1
                except Exception as e:
                    logger.error(f"Error saving file: {str(e)}")
                    continue
        
        if saved_count == 0:
            flash("Failed to save any images. Please use valid image files.", "error")
            return redirect("/register")
        
        flash(f"✓ Successfully uploaded {saved_count} images for {name}!", "success")
        return redirect("/register")
    
    return render_template("register.html")

@app.route("/capture_webcam", methods=["POST"])
def capture_webcam():
    """Trigger webcam capture for a person"""
    name = request.form.get("name", "").strip()
    
    if not name:
        flash("Please enter a name", "error")
        return redirect("/register")
    
    try:
        # Run capture.py with the name parameter
        result = subprocess.run(
            [sys.executable, "capture.py", name],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode == 0:
            flash(f"✓ Successfully captured images for {name}!", "success")
        else:
            if "No faces detected" in result.stdout or "No faces detected" in result.stderr:
                flash(f"No faces detected. Check lighting and camera position.", "error")
            else:
                flash(f"Capture completed with message: {result.stdout or result.stderr}", "info")
        
        return redirect("/register")
    except subprocess.TimeoutExpired:
        flash("Capture session timed out", "error")
        return redirect("/register")
    except Exception as e:
        logger.error(f"Error in webcam capture: {str(e)}")
        flash(f"Error: {str(e)}", "error")
        return redirect("/register")


@app.route('/register_webcam', methods=['POST'])
def register_webcam():
    """Endpoint to receive webcam-captured images from browser and save them to dataset."""
    try:
        name = request.form.get('name', '').strip()
        if not name:
            return jsonify({'status': 'error', 'message': 'Name is required'}), 400

        # Support files named images[] or images
        files = request.files.getlist('images[]') or request.files.getlist('images') or []

        person_path = Path('dataset') / name
        person_path.mkdir(parents=True, exist_ok=True)

        saved = 0
        for idx, f in enumerate(files):
            try:
                # Ensure we have a filename; browsers may send blob without filename
                fn = getattr(f, 'filename', None) or f'image_{idx}.jpg'
                # Save all uploads as jpg to avoid strict extension checks
                filename = secure_filename(f"{saved}.jpg")
                filepath = person_path / filename
                f.save(str(filepath))
                saved += 1
            except Exception as e:
                logger.warning(f"Failed to save uploaded frame #{idx}: {e}")
                continue

        if saved == 0:
            return jsonify({'status': 'error', 'message': 'No images saved. Ensure camera works and images are captured.'}), 400

        return jsonify({'status': 'ok', 'message': f'Saved {saved} images for {name}'}), 200

    except Exception as e:
        logger.error(f"register_webcam error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/get_users', methods=['GET'])
def get_users():
    """Get list of registered users from dataset folder"""
    try:
        dataset_path = Path('dataset')
        users = []
        
        if dataset_path.exists():
            for person_dir in sorted(dataset_path.glob('*/')):
                if person_dir.is_dir():
                    image_count = len(list(person_dir.glob('*.jpg')) + list(person_dir.glob('*.png')))
                    if image_count > 0:
                        users.append({
                            'name': person_dir.name,
                            'image_count': image_count
                        })
        
        return jsonify({'status': 'ok', 'users': users}), 200
    except Exception as e:
        logger.error(f"get_users error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/train_selected', methods=['POST'])
def train_selected():
    """Train model on selected users with progress updates"""
    try:
        data = request.get_json()
        selected_users = data.get('users', [])
        
        if not selected_users:
            return jsonify({'status': 'error', 'message': 'Please select at least one user'}), 400
        
        # Check if there are at least 2 users for SVM training
        if len(selected_users) < 2:
            return jsonify({'status': 'error', 'message': 'You need at least 2 different people to train the model. Please register and add images for more people.'}), 400
        
        dataset_path = Path('dataset')
        X, y = [], []
        processed = 0
        total_images = 0
        
        # First pass: count total images
        for username in selected_users:
            person_dir = dataset_path / username
            if person_dir.exists():
                images = list(person_dir.glob('*.jpg')) + list(person_dir.glob('*.png'))
                total_images += len(images)
        
        if total_images == 0:
            return jsonify({'status': 'error', 'message': 'No images found for selected users'}), 400
        
        detector = get_detector()
        embedder = get_embedder()
        
        logger.info(f"Starting training on {len(selected_users)} users with {total_images} images...")
        
        # Extract embeddings from selected users
        for username in selected_users:
            person_dir = dataset_path / username
            if not person_dir.exists():
                continue
            
            image_files = sorted(list(person_dir.glob('*.jpg')) + list(person_dir.glob('*.png')))
            
            for img_path in image_files:
                embedding = extract_embedding(img_path)
                if embedding is not None:
                    X.append(embedding)
                    y.append(username)
                
                processed += 1
                progress = int((processed / total_images) * 100)
                logger.info(f"Extracted {processed}/{total_images} embeddings ({progress}%)")
        
        if len(X) == 0:
            return jsonify({'status': 'error', 'message': 'No valid face embeddings found. Check image quality.'}), 400
        
        # Train SVM model
        logger.info("Training SVM model...")
        from sklearn.svm import SVC
        from sklearn.preprocessing import LabelEncoder
        
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        
        model = SVC(kernel='linear', probability=True, random_state=42)
        model.fit(X, y_encoded)
        
        # Save models
        models_path = Path('models')
        models_path.mkdir(exist_ok=True)
        
        with open(models_path / 'svm.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open(models_path / 'encoder.pkl', 'wb') as f:
            pickle.dump(encoder, f)
        
        logger.info(f"Training completed: {len(selected_users)} users, {len(X)} samples")
        return jsonify({
            'status': 'ok',
            'message': f'Model trained successfully! ({len(selected_users)} users, {len(X)} samples)'
        }), 200
        
    except Exception as e:
        logger.error(f"train_selected error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500



@app.route("/train", methods=["GET", "POST"])
def train():
    # Just serve the form - training is handled by /train_selected endpoint via AJAX
    return render_template("train.html")

@app.route("/attendance", methods=["GET", "POST"])
def attendance():
    if request.method == "POST":
        try:
            # Check if models exist
            models_path = Path("models")
            if not (models_path / "svm.pkl").exists() or not (models_path / "encoder.pkl").exists():
                flash("Models not trained yet. Please train the model first.", "error")
                return redirect("/attendance")
            
            # Load models
            with open(models_path / "svm.pkl", "rb") as f:
                svm = pickle.load(f)
            with open(models_path / "encoder.pkl", "rb") as f:
                encoder = pickle.load(f)
            
            flash("📌 Starting attendance capture. Check console for camera window.", "info")
            
            # Run attendance script
            result = subprocess.run(
                [sys.executable, "attendance.py"],
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode == 0:
                flash("✓ Attendance recording completed!", "success")
            else:
                flash(f"Attendance process completed. Check console for details.", "info")
            
            return redirect("/")
        except subprocess.TimeoutExpired:
            flash("Attendance session timed out", "error")
            return redirect("/attendance")
        except Exception as e:
            logger.error(f"Attendance error: {str(e)}")
            flash(f"Error: {str(e)}", "error")
            return redirect("/attendance")
    
    return render_template("attendance.html")

@app.route("/view")
def view():
    records = get_attendance_records()
    return render_template("view.html", data=records)

@app.errorhandler(404)
def not_found(error):
    return redirect("/")

@app.errorhandler(500)
def server_error(error):
    logger.error(f"Server error: {str(error)}")
    flash("An error occurred. Please try again.", "error")
    return redirect("/")

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)