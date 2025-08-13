import os
import cv2
import pickle
import signal
import numpy as np
import pandas as pd
from datetime import datetime
from deepface import DeepFace
from mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity
from retinaface import RetinaFace

# Graceful shutdown handler
def handler(signum, frame):
    print("\nShutting down gracefully...")
    raise KeyboardInterrupt

signal.signal(signal.SIGINT, handler)

# Load stored embeddings
embeddings_path = "face_embeddings.pkl"
if not os.path.exists(embeddings_path):
    print("Error: Face embeddings file not found. Run `train_face_recognizer.py` first.")
    exit()

with open(embeddings_path, "rb") as f:
    known_embeddings = pickle.load(f)

print(f"Loaded embeddings for {len(known_embeddings)} people: {list(known_embeddings.keys())}")

# Initialize face detectors
detector = MTCNN()
try:
    haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if haar_cascade.empty():
        raise ValueError("Haar Cascade classifier failed to load")
except Exception as e:
    print(f"Error loading Haar Cascade: {e}")
    haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    if haar_cascade.empty():
        print("Could not load Haar Cascade classifier")
        haar_cascade = None

def validate_face(img, box):
    """Validate if the detected region likely contains a real face"""
    x, y, w, h = box
    
    # Check aspect ratio
    aspect_ratio = h / w
    if not (0.6 <= aspect_ratio <= 1.6):  
        return False
    
    # Check size constraints
    if w < 30 or h < 30 or w > img.shape[1] * 0.95 or h > img.shape[0] * 0.95:
        return False
    
    # Check pixel variation
    face_region = img[y:y+h, x:x+w]
    if face_region.size == 0:
        return False
    
    gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    std_dev = np.std(gray_face)
    if std_dev < 15:  
        return False
    
    # Check skin tone
    hsv_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 10, 40])
    upper_skin = np.array([30, 180, 255])
    skin_mask = cv2.inRange(hsv_face, lower_skin, upper_skin)
    skin_ratio = np.sum(skin_mask > 0) / (w * h)
    if skin_ratio < 0.15:
        return False
    
    return True

def detect_faces(img):
    """Detect faces using multiple methods with fallbacks"""
    # Try MTCNN first
    try:
        faces = detector.detect_faces(img)
        valid_faces = [face for face in faces if face['confidence'] > 0.7 and validate_face(img, face['box'])]
        if valid_faces:
            return valid_faces, 'mtcnn'
    except Exception as e:
        print(f"MTCNN error: {e}")

    # Try Haar Cascade if available
    if haar_cascade is not None:
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            haar_faces = haar_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            valid_faces = [{"box": [x,y,w,h]} for (x,y,w,h) in haar_faces if validate_face(img, [x,y,w,h])]
            if valid_faces:
                return valid_faces, 'haar'
        except Exception as e:
            print(f"Haar Cascade error: {e}")

    # Try RetinaFace as last resort
    try:
        retina_results = RetinaFace.detect_faces(img, threshold=0.5)
        if isinstance(retina_results, dict):
            valid_faces = []
            for key in retina_results:
                face_info = retina_results[key]
                facial_area = face_info.get("facial_area", None)
                if facial_area:
                    x1, y1, x2, y2 = facial_area
                    box = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                    if validate_face(img, box):
                        valid_faces.append({"box": box})
            if valid_faces:
                return valid_faces, 'retinaface'
    except Exception as e:
        print(f"RetinaFace error: {e}")
    
    return [], 'none'

def recognize_face(face_img):
    """Recognize a face using DeepFace embeddings"""
    try:
        # Use the same model that was used to create the embeddings
        embedding = DeepFace.represent(
            img_path=face_img,
            model_name="ArcFace",  # Must match the model used to create embeddings
            enforce_detection=False
        )[0]["embedding"]
        
        best_match = "Unknown"
        best_score = 0.0

        for person, stored_embeddings in known_embeddings.items():
            # Ensure the stored embeddings have the same dimension
            if len(embedding) != len(stored_embeddings[0]):
                raise ValueError(f"Embedding dimension mismatch: Current {len(embedding)} vs Stored {len(stored_embeddings[0])}")
            
            scores = [cosine_similarity([embedding], [e])[0][0] for e in stored_embeddings]
            max_score = max(scores) if scores else 0.0
            if max_score > best_score:
                best_score = max_score
                best_match = person if max_score > 0.6 else "Unknown"

        return best_match, best_score * 100
    except Exception as e:
        print(f"Face recognition error: {e}")
        return "Unknown", 0.0

def get_attendance_filename():
    """Generate filename for today's attendance file"""
    date_str = datetime.now().strftime("%Y-%m-%d")
    return f"attendance_{date_str}.xlsx"

def mark_attendance(name, accuracy):
    """Mark attendance in Excel file with time validation"""
    timestamp = datetime.now()
    current_time = timestamp.strftime('%Y-%m-%d %H:%M:%S')
    excel_file = get_attendance_filename()

    new_entry = pd.DataFrame([[name, f"{accuracy:.2f}%", current_time]], 
                           columns=["Name", "Accuracy (%)", "Timestamp"])

    if os.path.exists(excel_file):
        df = pd.read_excel(excel_file)
        if not df.empty and name in df["Name"].values:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            last_entry_time = df[df["Name"] == name]["Timestamp"].max()
            
            if (timestamp - last_entry_time).total_seconds() < 1800:  
                print(f"{name} was already marked present {(timestamp - last_entry_time).seconds//60} minutes ago")
                return
            
        df = pd.concat([df, new_entry], ignore_index=True)
    else:
        df = new_entry

    df.to_excel(excel_file, index=False)
    print(f"Attendance marked for {name} (Accuracy: {accuracy:.2f}%)")

def recognize_faces_in_image(image_path):
    """Recognize faces in a static image"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read the image file")

        faces, detection_method = detect_faces(img)
        if not faces:
            print(f"No faces detected using {detection_method}")
            return

        for face in faces:
            x, y, w, h = face["box"]
            x2, y2 = x + w, y + h

            face_crop = img[y:y2, x:x2]
            cv2.imwrite("temp_face.jpg", face_crop)
            recognized_name, accuracy = recognize_face("temp_face.jpg")
            
            cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{recognized_name} ({accuracy:.2f}%)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if recognized_name != "Unknown":
                mark_attendance(recognized_name, accuracy)

        cv2.imshow("Recognized Faces", img)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Image processing error: {e}")

def recognize_faces_in_video(video_path):
    """Recognize faces in a video file"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        marked_attendance = set()
        frame_skip = 2  # Process every other frame
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            faces, _ = detect_faces(frame)
            for face in faces:
                x, y, w, h = face["box"]
                x2, y2 = x + w, y + h

                face_crop = frame[y:y2, x:x2]
                cv2.imwrite("temp_face.jpg", face_crop)
                recognized_name, accuracy = recognize_face("temp_face.jpg")

                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{recognized_name} ({accuracy:.2f}%)", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                if recognized_name != "Unknown" and recognized_name not in marked_attendance:
                    mark_attendance(recognized_name, accuracy)
                    marked_attendance.add(recognized_name)

            cv2.imshow("Video Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Video processing error: {e}")

def recognize_faces_from_webcam():
    """Live face recognition from webcam"""
    try:
        # Try different camera indices
        for camera_index in [0, 1]:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                print(f"Using camera {camera_index}")
                break
        else:
            raise ValueError("Could not access any webcam")

        # Create directory for detected faces
        detected_faces_dir = "detected_faces"
        os.makedirs(detected_faces_dir, exist_ok=True)

        marked_attendance = set()
        frame_skip = 3  # Process every 3rd frame
        frame_count = 0
        last_recognition = {"name": "Unknown", "score": 0.0, "frame": None, "face_crop": None}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            # Resize frame for faster processing
            frame = cv2.resize(frame, (0,0), fx=0.7, fy=0.7)

            faces, _ = detect_faces(frame)
            for face in faces:
                x, y, w, h = face["box"]
                x2, y2 = x + w, y + h

                face_crop = frame[y:y2, x:x2]
                cv2.imwrite("temp_face.jpg", face_crop)
                recognized_name, accuracy = recognize_face("temp_face.jpg")

                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{recognized_name} ({accuracy:.2f}%)", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if recognized_name != "Unknown":
                    if recognized_name not in marked_attendance:
                        mark_attendance(recognized_name, accuracy)
                        marked_attendance.add(recognized_name)
                    
                    if accuracy > last_recognition["score"]:
                        last_recognition = {
                            "name": recognized_name,
                            "score": accuracy,
                            "frame": frame.copy(),
                            "face_crop": face_crop.copy()
                        }

            cv2.imshow("Live Face Recognition", frame)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break

        # Save last detected face
        if last_recognition["frame"] is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            frame_filename = os.path.join(detected_faces_dir, f"{last_recognition['name']}_{timestamp}_frame.jpg")
            face_filename = os.path.join(detected_faces_dir, f"{last_recognition['name']}_{timestamp}_face.jpg")
            
            cv2.imwrite(frame_filename, last_recognition["frame"])
            cv2.imwrite(face_filename, last_recognition["face_crop"])
            
            print(f"\nSaved last detection: {face_filename}")

        cap.release()
        cv2.destroyAllWindows()
        
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Webcam error: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        print("\nFace Recognition System")
        print("1. Recognize faces from image")
        print("2. Live webcam recognition")
        print("3. Recognize faces from video")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            image_path = input("Enter image path: ").strip()
            recognize_faces_in_image(image_path)
        elif choice == "2":
            recognize_faces_from_webcam()
        elif choice == "3":
            video_path = input("Enter video path: ").strip()
            recognize_faces_in_video(video_path)
        else:
            print("Invalid choice")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()