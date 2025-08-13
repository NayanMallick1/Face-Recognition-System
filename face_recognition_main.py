import os
import cv2
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from deepface import DeepFace
from mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity
from retinaface import RetinaFace

# Load stored embeddings
embeddings_path = "face_embeddings0.pkl"
if not os.path.exists(embeddings_path):
    print("Error: Face embeddings file not found. Run `train_face_recognizer.py` first.")
    exit()

with open(embeddings_path, "rb") as f:
    known_embeddings = pickle.load(f)

print(f"Loaded embeddings for {len(known_embeddings)} people: {list(known_embeddings.keys())}")

# Initialize face detectors
detector = MTCNN()
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def validate_face(img, box):
    """Validate if the detected region likely contains a real face using multiple checks"""
    x, y, w, h = box
    
    # Check aspect ratio (typical face aspect ratio is between 0.6 and 1.6)
    aspect_ratio = h / w
    if not (0.6 <= aspect_ratio <= 1.6):  
        return False
    
    # Check minimum and maximum size - more relaxed
    if w < 30 or h < 30 or w > img.shape[1] * 0.95 or h > img.shape[0] * 0.95:
        return False
    
    # Check if the region has enough variation in pixel values
    face_region = img[y:y+h, x:x+w]
    if face_region.size == 0:
        return False
    
    # Convert to grayscale for texture analysis
    gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    
    # Check standard deviation (texture variation) - reduced threshold
    std_dev = np.std(gray_face)
    if std_dev < 15:  
        return False
    
    # Check for skin tone range in HSV color space - broader range
    hsv_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
    # Define skin tone range in HSV (even broader range)
    lower_skin = np.array([0, 10, 40])     # More relaxed lower bounds
    upper_skin = np.array([30, 180, 255])  # More relaxed upper bounds
    skin_mask = cv2.inRange(hsv_face, lower_skin, upper_skin)
    skin_ratio = np.sum(skin_mask > 0) / (w * h)
    if skin_ratio < 0.15:  # Reduced threshold to 15% skin tone
        return False
    
    # Check for symmetry (faces are typically symmetrical) - more relaxed
    left_half = gray_face[:, :w//2]
    right_half = cv2.flip(gray_face[:, w//2:], 1)
    if min(left_half.shape[1], right_half.shape[1]) > 0:
        symmetry_score = np.mean(np.abs(left_half[:, :min(left_half.shape[1], right_half.shape[1])] - 
                                      right_half[:, :min(left_half.shape[1], right_half.shape[1])]))
        if symmetry_score > 85:  # More relaxed symmetry threshold
            return False
    
    return True

# Helper: Detect faces using multiple methods
def detect_faces(img):
    # Try MTCNN first
    faces = detector.detect_faces(img)
    valid_faces = []
    
    if faces:
        for face in faces:
            if face['confidence'] > 0.01:  # Lowered confidence threshold significantly
                box = face['box']
                if validate_face(img, box):
                    valid_faces.append(face)
    
    if valid_faces:
        return valid_faces, 'mtcnn'
    
    # Try RetinaFace next
    try:
        retina_results = RetinaFace.detect_faces(img, threshold=0.7)
        if isinstance(retina_results, dict):
            valid_faces_retina = []
            for key in retina_results:
                face_info = retina_results[key]
                facial_area = face_info.get("facial_area", None)
                if facial_area:
                    x1, y1, x2, y2 = facial_area
                    box = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                    if validate_face(img, box):
                        valid_faces_retina.append({"box": box})
            if valid_faces_retina:
                return valid_faces_retina, 'retinaface'
    except Exception as e:
        print(f"RetinaFace error: {e}")
    
    # If both fail, try Haar Cascade
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Equalize histogram to improve detection in different lighting
    gray = cv2.equalizeHist(gray)
    
    haar_faces = haar_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,     # More aggressive scaling
        minNeighbors=4,      # Reduced for more detections
        minSize=(60,60),    # Smaller minimum face size
        maxSize=(int(img.shape[1] * 0.95), int(img.shape[0] * 0.95)),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    valid_haar_faces = []
    for (x, y, w, h) in haar_faces:
        valid_haar_faces.append({"box": [x, y, w, h]})
    
    if valid_haar_faces:
        return valid_haar_faces, 'haar'
    
    return [], 'none'

# Function to recognize a face from an image file (using DeepFace with ArcFace)
def recognize_face(face_img):
    try:
        # Set enforce_detection to False to avoid DeepFace's internal detection
        embedding = DeepFace.represent(img_path=face_img, model_name="ArcFace", enforce_detection=False)[0]["embedding"]
        best_match = "Unknown"
        best_score = 0.0

        # Compare the input embedding with each stored embedding
        for person, stored_embeddings in known_embeddings.items():
            scores = [cosine_similarity([embedding], [stored_embedding])[0][0] for stored_embedding in stored_embeddings]
            max_score = max(scores) if scores else 0.0
            print(f"Similarity with {person}: {max_score:.2f}")
            if max_score > best_score:
                best_score = max_score
                best_match = person if max_score > 0.65 else "Unknown"  # Lowered threshold

        return best_match, best_score * 100
    except Exception as e:
        print(f"Face recognition error: {e}")
        return "Unknown", 0.0

# Helper: Generate a filename for today's attendance Excel file
def get_attendance_filename():
    date_str = datetime.now().strftime("%Y-%m-%d")
    return f"attendance_{date_str}.xlsx"

# Function to mark attendance in an Excel file (one entry per candidate per session)
def mark_attendance(name, accuracy):
    timestamp = datetime.now()
    current_time = timestamp.strftime('%Y-%m-%d %H:%M:%S')
    excel_file = get_attendance_filename()

    new_entry = pd.DataFrame([[name, f"{accuracy:.2f}%", current_time]], 
                           columns=["Name", "Accuracy (%)", "Timestamp"])

    if os.path.exists(excel_file):
        df = pd.read_excel(excel_file)
        if not df.empty and name in df["Name"].values:
            # Convert timestamp strings to datetime objects for comparison
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            last_entry_time = df[df["Name"] == name]["Timestamp"].max()
            
            # Check if enough time has passed (30 minutes) since last entry
            time_diff = timestamp - last_entry_time
            if time_diff.total_seconds() < 1800:  
                print(f"{name} was already marked present {time_diff.seconds//60} minutes ago. Minimum interval is 30 minutes.")
                return
            
        df = pd.concat([df, new_entry], ignore_index=True)
    else:
        df = new_entry

    df.to_excel(excel_file, index=False)
    print(f"Attendance marked for {name} (Accuracy: {accuracy:.2f}%)")

# Function to recognize faces from an image file
def recognize_faces_in_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Could not read the image file.")
        return

    faces, detection_method = detect_faces(img)
    if not faces:
        print(f"No face detected in the image using {detection_method}.")
        return

    for face in faces:
        x, y, width, height = face["box"]
        x2, y2 = x + width, y + height

        # Crop and save temporary face image
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

# Function to recognize faces from a video file (attendance marked once per session)
def recognize_faces_in_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open the video file.")
        return

    marked_attendance = set()  

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        faces, detection_method = detect_faces(frame)
        for face in faces:
            x, y, width, height = face["box"]
            x2, y2 = x + width, y + height

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

# Function to recognize faces from the live webcam (attendance marked once per session)
def recognize_faces_from_webcam():
    # Try different camera indices
    for camera_index in [0, 1]:
        print(f"Trying to access camera {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            print(f"Successfully opened camera {camera_index}")
            break
    else:
        print("Could not access any webcam. Please check your camera connection.")
        return

    # Create directory for saving detected faces if it doesn't exist
    detected_faces_dir = "detected_faces"
    if not os.path.exists(detected_faces_dir):
        os.makedirs(detected_faces_dir)

    marked_attendance = set()
    frame_count = 0
    last_recognition = {"name": "Unknown", "score": 0.0, "frame": None, "face_crop": None}
    should_show_last_face = True  # Flag to control last face display

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera")
            break

        frame_count += 1
        if frame_count % 2 != 0:  # Process every other frame to reduce CPU load
            continue

        faces, detection_method = detect_faces(frame)
        print(f"Detected {len(faces)} faces using {detection_method}")
        
        for face in faces:
            x, y, width, height = face["box"]
            x2, y2 = x + width, y + height

            # Ensure coordinates are within frame bounds
            x, y = max(0, x), max(0, y)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            if x2 <= x or y2 <= y:
                continue

            face_crop = frame[y:y2, x:x2]
            if face_crop.size == 0:
                continue

            cv2.imwrite("temp_face.jpg", face_crop)
            recognized_name, accuracy = recognize_face("temp_face.jpg")

            # Draw rectangle and name
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{recognized_name} ({accuracy:.2f}%)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Update last recognition if this is a known person with higher confidence
            if recognized_name != "Unknown" and accuracy > last_recognition["score"]:
                last_recognition = {
                    "name": recognized_name,
                    "score": accuracy,
                    "frame": frame.copy(),
                    "face_crop": face_crop.copy()
                }

            if recognized_name != "Unknown" and recognized_name not in marked_attendance:
                mark_attendance(recognized_name, accuracy)
                marked_attendance.add(recognized_name)

        cv2.imshow("Live Face Recognition", frame)
        
        # Check for 'q' key press to quit
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            should_show_last_face = False  # Skip showing last face
            break

    print("Cleaning up...")
    
    # Only show last face if we didn't press 'q' to exit
    if should_show_last_face and last_recognition["frame"] is not None:
        print(f"\nLast detected person: {last_recognition['name']} with confidence: {last_recognition['score']:.2f}%")
        
        # Save the full frame
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        frame_filename = os.path.join(detected_faces_dir, f"{last_recognition['name']}_{timestamp}_frame.jpg")
        cv2.imwrite(frame_filename, last_recognition["frame"])
        
        # Save the face crop
        face_filename = os.path.join(detected_faces_dir, f"{last_recognition['name']}_{timestamp}_face.jpg")
        cv2.imwrite(face_filename, last_recognition["face_crop"])
        
        print(f"Saved detected face to: {face_filename}")
        print(f"Saved full frame to: {frame_filename}")
        
        # Show the images
        cv2.imshow("Last Detected Face (Full Frame)", last_recognition["frame"])
        cv2.imshow("Last Detected Face (Cropped)", last_recognition["face_crop"])
        cv2.waitKey(0)  # Wait until user presses a key
    
    cap.release()
    cv2.destroyAllWindows()

# Main menu
if __name__ == "__main__":
    print("\nSelect an option:")
    print("1. Recognize a face from an image")
    print("2. Start live face recognition (Webcam)")
    print("3. Recognize a face from a video file")

    choice = input("Enter your choice (1, 2, or 3): ").strip()

    if choice == "1":
        image_path = input("Enter the image file path: ").strip()
        recognize_faces_in_image(image_path)
    elif choice == "2":
        recognize_faces_from_webcam()
    elif choice == "3":
        video_path = input("Enter the video file path: ").strip()
        recognize_faces_in_video(video_path)
    else:
        print("Invalid choice. Please restart the program.")