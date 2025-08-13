import os
import cv2
import pickle
import json
import numpy as np
import pandas as pd
from deepface import DeepFace

# Dataset and model storage
dataset_path = "faceData"
embeddings_path = "face_embeddings0.pkl"
metadata_path = "processed_images0.json"  # ✅ Track processed images

# ✅ Load existing embeddings and processed images
def load_existing_data():
    if os.path.exists(embeddings_path):
        with open(embeddings_path, "rb") as f:
            embeddings = pickle.load(f)
    else:
        embeddings = {}

    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            processed_images = json.load(f)
    else:
        processed_images = {}

    return embeddings, processed_images

# ✅ Function to update embeddings incrementally
def store_face_embeddings():
    embeddings, processed_images = load_existing_data()  # Load previous data

    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)

        if os.path.isdir(person_path):
            if person_name not in embeddings:
                embeddings[person_name] = []  # Add new person

            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)

                # ✅ Check if the image was already processed
                if person_name in processed_images and image_name in processed_images[person_name]:
                    print(f"🔄 Skipped already processed: {image_name} for {person_name}")
                    continue  # Skip already processed images

                try:
                    # ✅ Extract face embeddings for new images only
                    embedding = DeepFace.represent(img_path=image_path, model_name="ArcFace")[0]["embedding"]
                    embeddings[person_name].append(embedding)

                    # ✅ Mark this image as processed
                    if person_name not in processed_images:
                        processed_images[person_name] = []
                    processed_images[person_name].append(image_name)

                    print(f"✅ Processed new image: {image_name} for {person_name}")

                except Exception as e:
                    print(f"❌ Error processing {image_name}: {e}")

    # ✅ Save updated embeddings
    with open(embeddings_path, "wb") as f:
        pickle.dump(embeddings, f)

    # ✅ Save metadata of processed images
    with open(metadata_path, "w") as f:
        json.dump(processed_images, f, indent=4)

    print("✅ Face embeddings updated successfully!")
    generate_candidates_excel()

def generate_candidates_excel():
    """
    Generates an Excel file listing each candidate's name and folder.
    """
    candidate_names = []
    candidate_folders = []
    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)
        if os.path.isdir(person_path):
            candidate_names.append(person)
            candidate_folders.append(person)  # Here folder name equals the person name
    df = pd.DataFrame({"Name": candidate_names, "Folder": candidate_folders})
    excel_filename = "candidates.xlsx"
    df.to_excel(excel_filename, index=False)
    print(f"✅ Candidates Excel file generated: {excel_filename}")

if __name__ == "__main__":
    store_face_embeddings()
