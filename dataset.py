# Install dependencies if not already installed:
# pip install google_images_download mtcnn opencv-python pillow

import os
import cv2
from mtcnn import MTCNN
from google_images_download import google_images_download

def download_images(person_name, limit, output_dir):
    response = google_images_download.googleimagesdownload()
    arguments = {
        "keywords": person_name,
        "limit": limit,
        "print_urls": False,
        "format": "jpg",
        "size": "medium",
        "output_directory": output_dir,
        "no_directory": True  # save directly into folder
    }
    response.download(arguments)

def crop_and_align_faces(input_folder, output_folder, image_size=256):
    os.makedirs(output_folder, exist_ok=True)
    detector = MTCNN()
    
    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        try:
            image = cv2.imread(img_path)
            faces = detector.detect_faces(image)
            if len(faces) == 0:
                continue  # skip if no face detected
            
            x, y, w, h = faces[0]['box']
            x, y = max(0, x), max(0, y)
            face = image[y:y+h, x:x+w]
            face = cv2.resize(face, (image_size, image_size))
            
            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path, face)
            print(f"[+] Saved aligned face: {save_path}")
        except Exception as e:
            print(f"[!] Failed {filename}: {e}")

def prepare_dataset(person_list, dataset_type, num_images=50):
    """
    person_list: list of names
    dataset_type: 'src' or 'dst'
    """
    for person_name in person_list:
        print(f"\nDownloading images for {person_name} ({dataset_type})...")
        raw_dir = os.path.join("dataset", f"{dataset_type}_raw", person_name.replace(" ", "_"))
        output_dir = os.path.join("dataset", f"{dataset_type}_images", person_name.replace(" ", "_"))
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        download_images(person_name, num_images, raw_dir)
        crop_and_align_faces(raw_dir, output_dir)

if __name__ == "__main__":
    # ======= USER SETTINGS =======
    src_persons = ["Man smiling", "Woman smiling", "Young adult male"]  
    dst_persons = ["Man neutral", "Woman neutral", "Older adult male"]  
    num_images_per_person = 50
    # ==============================
    
    # Prepare source dataset
    prepare_dataset(src_persons, dataset_type="src", num_images=num_images_per_person)
    
    # Prepare target dataset
    prepare_dataset(dst_persons, dataset_type="dst", num_images=num_images_per_person)
    
    print("\nMulti-person generic dataset preparation complete!")
