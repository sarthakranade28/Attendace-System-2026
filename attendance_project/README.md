# Face Attendance System

A professional facial recognition-based attendance system built with Flask, FaceNet, and MTCNN. The system automatically detects faces and marks attendance in real-time.

## Features

✅ **Real-time Face Detection** - Uses MTCNN for accurate face detection  
✅ **AI-Powered Recognition** - FaceNet for advanced face embeddings  
✅ **SQLite Database** - Efficient and secure attendance storage  
✅ **Professional Web UI** - Bootstrap 5 responsive interface  
✅ **Error Handling** - Comprehensive logging and error management  
✅ **Model Caching** - FaceNet automatically caches the pre-trained model  
✅ **Export Functionality** - Download attendance records as CSV  

## System Requirements

- Python 3.8+
- Webcam/Camera
- Windows/Mac/Linux
- Minimum 4GB RAM

## Installation

### 1. Clone or Extract Project
```bash
cd attendance_project
```

### 2. Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```

Then open your browser and go to: **http://127.0.0.1:5000**

## Project Structure

```
attendance_project/
├── app.py                 # Flask application (main server)
├── attendance.py          # Attendance capture logic
├── capture.py             # Face capture for registration
├── train.py               # Model training script
├── facenet.py             # FaceNet model utilities
├── requirements.txt       # Python dependencies
├── static/
│   └── style.css          # Professional styling
├── templates/
│   ├── index.html         # Home page
│   ├── register.html      # Student registration
│   ├── attendance.html    # Attendance instructions
│   └── view.html          # View attendance records
├── dataset/               # Captured face images (auto-created)
├── models/                # Trained models (auto-created)
└── attendance.db          # SQLite database (auto-created)
```

## Usage Guide

### Step 1: Register Students

1. Click **Register Student** on the home page
2. Enter the student's full name
3. Position face in camera frame
4. System captures 30 images automatically
5. Press ESC to stop early if needed

**Tips for Better Registration:**
- Ensure good lighting
- Face should be clear and visible
- Try different angles
- Eyes must be visible
- No sunglasses or obstructions

### Step 2: Train the Model

1. Click **Train Model** button
2. System processes all captured images
3. Extracts facial embeddings using FaceNet
4. Trains SVM classifier
5. Saves models to `models/` directory

**Training Tips:**
- Register at least 2-3 students before training
- Wait for training to complete (don't interrupt)
- Models are saved: `svm.pkl` and `encoder.pkl`

### Step 3: Take Attendance

1. Click **Take Attendance**
2. Click **Start Attendance Capture**
3. Position faces in camera frame
4. System automatically marks recognized faces
5. Press ESC to stop

**Attendance Tips:**
- Ensure good lighting
- Keep face centered
- Face the camera directly
- System requires minimum 70% confidence

### Step 4: View Records

1. Click **View Records**
2. See all attendance entries with timestamps
3. View statistics (total records, unique students, today's attendance)
4. **Export to CSV** for reports

## Database Schema

### Attendance Table
```sql
CREATE TABLE attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
```

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Web Framework | Flask 2.3.2 |
| Face Detection | MTCNN |
| Face Recognition | FaceNet (Keras) |
| Machine Learning | Scikit-Learn (SVM) |
| Database | SQLite3 |
| Frontend | Bootstrap 5 |
| Computer Vision | OpenCV |

## Key Functions

### app.py
- **run_python_script()** - Safely execute Python scripts
- **get_attendance_records()** - Fetch from SQLite database
- **init_db()** - Initialize database schema

### capture.py
- **capture_faces()** - Capture face images for a person
- Validates name input
- Handles camera errors

### train.py
- **train_model()** - Train SVM classifier
- Extracts embeddings using FaceNet
- Saves trained models

### attendance.py
- **take_attendance()** - Real-time attendance capture
- **check_models_exist()** - Verify models before starting
- Marks faces with high confidence (>70%)

### facenet.py
- **get_facenet_model()** - Cached FaceNet model loading
- **get_embedding()** - Extract facial embeddings
- **clear_cache()** - Clear model cache if needed

## Troubleshooting

### Camera Not Opening
```
Error: Could not access camera
```
**Solution:**
- Check if camera is plugged in
- Ensure no other application is using the camera
- Grant camera permissions in system settings
- Try a different USB port

### Model Training Fails
```
Error: No dataset found
```
**Solution:**
- Register at least one student first
- Check `dataset/` folder has subdirectories
- Ensure images are in `.jpg` format
- Check file permissions

### Low Recognition Accuracy
**Solution:**
- Register more students (30+ face samples each)
- Use consistent lighting
- Capture images from different angles
- Check for good image quality
- Retrain the model after adding more data

### AttributeError: FaceNet
**Solution:**
- FaceNet will auto-download on first use (~170MB)
- Ensure internet connection during first training
- Model is cached in `~/.keras/models/`

## Configuration

### Confidence Threshold
Edit in `attendance.py` (line ~120):
```python
if prob > 0.7:  # Increase for stricter matching
```

### Number of Face Samples
Edit in `capture.py`:
```python
def capture_faces(name, num_samples=30):  # Change 30
```

### Training Parameters
Edit in `train.py`:
```python
model = SVC(kernel='linear', probability=True, random_state=42)
```

## Security Notes

⚠️ **Important:**
- Change `app.secret_key` in production
- Use HTTPS in production
- Protect the `models/` directory
- Backup `attendance.db` regularly
- Don't share trained models publicly

## Performance Tips

1. **Fast Registration:**
   - Good lighting reduces capture time
   - Only need 20-25 good quality images

2. **Fast Training:**
   - Register 5-10 students max for quick training
   - SVM trains quickly (~1-2 seconds per person)

3. **Fast Recognition:**
   - Fresh trained model: better accuracy
   - Consistent lighting: faster detection
   - Clear faces: instant recognition

## Known Limitations

- Single GPU/CPU processing
- Real-time only (no video files)
- Requires stable webcam
- Internet needed for first FaceNet download
- Limited to ~100 people practical limit

## Future Enhancements

- [ ] Multi-camera support
- [ ] Mobile app integration
- [ ] Advanced analytics dashboard
- [ ] Automated weekly reports
- [ ] Attendance predictions
- [ ] API for third-party integrations

## License

This project is open-source and available for educational purposes.

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Verify all dependencies are installed
3. Check camera permissions
4. Review error logs in console

## Version History

- **v1.0** (2026-03-17) - Initial release with SQLite integration
  - Added professional UI with Bootstrap 5
  - Added comprehensive error handling
  - Added model caching
  - Added export functionality

---

**Created with ❤️ for smarter attendance tracking**
