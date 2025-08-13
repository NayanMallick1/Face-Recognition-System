import os
import cv2
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from deepface import DeepFace
from mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity

# Load stored embeddings
embeddings_path = "face_embeddings.pkl"
excel_file = "attendance.xlsx"  

if not os.path.exists(embeddings_path):
    print("❌ Error: Face embeddings file not found. Run `train_face_recognizer.py` first.")
    exit()

with open(embeddings_path, "rb") as f:
    known_embeddings = pickle.load(f)

# Initialize MTCNN detector
detector = MTCNN()

# Function to recognize faces
def recognize_face(face_img):
    try:
        embedding = DeepFace.represent(img_path=face_img, model_name="ArcFace")[0]["embedding"]
        best_match = "Unknown"
        best_score = 0.0

        for person, stored_embeddings in known_embeddings.items():
            scores = [cosine_similarity([embedding], [stored_embedding])[0][0] for stored_embedding in stored_embeddings]
            max_score = max(scores) if scores else 0.0
            
            if max_score > best_score:
                best_score = max_score
                best_match = person if best_score > 0.7 else "Unknown"

        return best_match, best_score * 100
    except Exception as e:
        print(f"❌ Face recognition error: {e}")
        return "Unknown", 0.0

# ✅ Function to mark attendance in Excel
def mark_attendance(name, accuracy):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    new_entry = pd.DataFrame([[name, f"{accuracy:.2f}%", timestamp]], columns=["Name", "Accuracy (%)", "Timestamp"])

    if os.path.exists(excel_file):
        df = pd.read_excel(excel_file)
        df = pd.concat([df, new_entry], ignore_index=True)  
    else:
        df = new_entry

    df.to_excel(excel_file, index=False)
    print(f"✅ Attendance marked for {name} (Accuracy: {accuracy:.2f}%)")

# ✅ Function to recognize faces from an image
def recognize_faces_in_image(image_path):
    img = cv2.imread(image_path)
    faces = detector.detect_faces(img)

    for face in faces:
        x, y, width, height = face["box"]
        x2, y2 = x + width, y + height

        face_crop = img[y:y2, x:x2]
        cv2.imwrite("temp_face.jpg", face_crop)

        recognized_name, accuracy = recognize_face("temp_face.jpg")

        cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{recognized_name} ({accuracy:.2f}%)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if recognized_name != "Unknown":
            mark_attendance(recognized_name, accuracy)

    cv2.imshow("Recognized Faces", img)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

# ✅ Function to recognize faces from a video file
def recognize_faces_in_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect_faces(frame)

        for face in faces:
            x, y, width, height = face["box"]
            x2, y2 = x + width, y + height

            face_crop = frame[y:y2, x:x2]
            cv2.imwrite("temp_face.jpg", face_crop)

            recognized_name, accuracy = recognize_face("temp_face.jpg")

            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{recognized_name} ({accuracy:.2f}%)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if recognized_name != "Unknown":
                mark_attendance(recognized_name, accuracy)

        cv2.imshow("Video Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ✅ Function to recognize faces in real-time webcam
def recognize_faces_from_webcam():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect_faces(frame)

        for face in faces:
            x, y, width, height = face["box"]
            x2, y2 = x + width, y + height

            face_crop = frame[y:y2, x:x2]
            cv2.imwrite("temp_face.jpg", face_crop)

            recognized_name, accuracy = recognize_face("temp_face.jpg")

            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{recognized_name} ({accuracy:.2f}%)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if recognized_name != "Unknown":
                mark_attendance(recognized_name, accuracy)

        cv2.imshow("Live Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ✅ Main menu
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
        print("❌ Invalid choice. Please restart the program.")
