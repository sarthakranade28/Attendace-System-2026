# Set TensorFlow environment variables BEFORE any TF/Keras imports
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import sys
from pathlib import Path
from mtcnn import MTCNN
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create dataset directory
Path("dataset").mkdir(exist_ok=True)

def capture_faces_from_webcam(name, num_samples=30):
    """
    Capture face images from webcam for dataset
    
    Args:
        name (str): Name of the person
        num_samples (int): Number of face samples to capture
    """
    # Validate name
    name = name.strip()
    if not name:
        logger.error("Invalid name provided")
        return False
    
    # Create person-specific directory
    path = Path("dataset") / name
    
    # Check if person already exists
    if path.exists() and list(path.glob("*.jpg")):
        print(f"\n⚠ '{name}' already has images in dataset")
        response = input("Do you want to overwrite? (y/n): ").lower()
        if response != 'y':
            logger.info("Capture cancelled")
            return False
        # Clear existing images
        for img_file in path.glob("*.jpg"):
            img_file.unlink()
            logger.info(f"Deleted: {img_file}")
    
    path.mkdir(parents=True, exist_ok=True)
    
    try:
        print("\n" + "="*60)
        print(f"📷 Starting Face Capture for: {name}")
        print("="*60)
        
        # Initialize detector
        print("Loading face detection model...")
        detector = MTCNN()
        
        # Initialize webcam
        print("Opening webcam...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            logger.error("Could not open camera")
            print("❌ Error: Could not open camera")
            print("\nTroubleshooting:")
            print("  - Check if camera is connected")
            print("  - Check if another app is using the camera")
            print("  - Try a different USB port")
            print("  - Restart the application")
            return False
        
        # Set camera properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        count = 0
        frame_count = 0
        detected_faces = 0
        
        print(f"\n📸 Instructions:")
        print(f"  • Position your face in the center of the frame")
        print(f"  • System will capture {num_samples} images automatically")
        print(f"  • Different angles and expressions = better accuracy")
        print(f"  • Press 'ESC' to stop capturing early")
        print(f"  • Press 'SPACE' to skip current frame")
        print(f"\n{'Target: ' + str(num_samples) + ' images'}\n")
        
        while count < num_samples:
            ret, frame = cap.read()
            frame_count += 1
            
            if not ret:
                logger.error("Failed to read frame from camera")
                print("❌ Failed to read from camera")
                break
            
            # Mirror the frame horizontally for selfie-like experience
            frame = cv2.flip(frame, 1)
            
            # Detect faces
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb)
            
            # Display face detection boxes and info
            for face in faces:
                x, y, w, h = face['box']
                confidence = face['confidence']
                
                # Draw bounding box
                color = (0, 255, 0) if confidence > 0.95 else (255, 165, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw confidence level
                conf_text = f"Confidence: {confidence:.2f}"
                cv2.putText(frame, conf_text, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Save face image on high confidence
                if confidence > 0.95:
                    face_img = frame[y:y+h, x:x+w]
                    if face_img.size > 0:
                        cv2.imwrite(str(path / f"{count}.jpg"), face_img)
                        detected_faces += 1
                        count += 1
                        
                        # Green highlight for saved image
                        cv2.rectangle(frame, (x-5, y-5), (x+w+5, y+h+5), (0, 255, 0), 3)
                        saved_text = f"✓ Saved: {count}/{num_samples}"
                        cv2.putText(frame, saved_text, (x, y + h + 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display status bar
            status_text = f"Captured: {count}/{num_samples} | Frame: {frame_count} | Detected: {len(faces)}"
            cv2.putText(frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display instructions
            instructions = "Press ESC to stop | SPACE to skip frame"
            cv2.putText(frame, instructions, (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
            
            # Show frame
            cv2.imshow(f"Capturing faces for {name} - {count}/{num_samples}", frame)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print(f"\n⏹ Capture stopped by user")
                break
            elif key == ord(' '):  # SPACE - skip frame
                print(f"⊘ Frame skipped")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        if count >= num_samples:
            success_msg = f"✓ Successfully captured {count} face samples for '{name}'"
            logger.info(success_msg)
            print(success_msg)
            print(f"  • Total frames: {frame_count}")
            print(f"  • Saved images: {count}")
            print(f"  • Efficiency: {(count/frame_count)*100:.1f}%")
            print("="*60)
            return True
        else:
            if count > 0:
                warning_msg = f"⚠ Only captured {count}/{num_samples} faces"
                logger.warning(warning_msg)
                print(warning_msg)
                print("  Tip: Better lighting and positioning helps!")
                print("="*60)
                return True  # Allow training with partial data
            else:
                error_msg = "❌ No faces detected. Check lighting and camera"
                logger.error(error_msg)
                print(error_msg)
                print("  Tips:")
                print("  • Ensure good lighting")
                print("  • Face should be clearly visible")
                print("  • Position face in center of frame")
                print("="*60)
                return False
            
    except Exception as e:
        error_msg = f"Error during face capture: {str(e)}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        return False

if __name__ == "__main__":
    # Get name from command line argument or user input
    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        print("\n" + "="*60)
        print("Face Registration System")
        print("="*60)
        name = input("Enter the person's full name: ")
    
    if not name.strip():
        print("Error: Name cannot be empty")
        sys.exit(1)
    
    success = capture_faces_from_webcam(name)
    sys.exit(0 if success else 1)