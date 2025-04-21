import cv2
import dlib
import numpy as np
import os
from tqdm import tqdm

def get_face_landmarks(image, predictor):
    """Detect facial landmarks using dlib"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray)
    
    if len(faces) != 1:
        return None
        
    landmarks = predictor(gray, faces[0])
    return np.array([[p.x, p.y] for p in landmarks.parts()])

def align_face(image_path, predictor, output_size=256):
    """Align face using facial landmarks"""
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    landmarks = get_face_landmarks(img, predictor)
    if landmarks is None:
        return None
        
    # Calculate transformation matrix
    # (Implementation of similarity transform based on landmarks)
    # ... [actual alignment code] ...
    
    aligned = cv2.warpAffine(img, transformation_matrix, (output_size, output_size))
    return aligned

def filter_dataset(input_dir, output_dir, predictor_path):
    predictor = dlib.shape_predictor(predictor_path)
    os.makedirs(output_dir, exist_ok=True)
    
    for img_name in tqdm(os.listdir(input_dir)):
        if not img_name.endswith(".chip.jpg"):
            continue
            
        img_path = os.path.join(input_dir, img_name)
        aligned = align_face(img_path, predictor)
        
        if aligned is not None:
            # Save with same name but without .chip.jpg
            output_name = img_name.replace('.chip.jpg', '.jpg')
            output_path = os.path.join(output_dir, output_name)
            cv2.imwrite(output_path, aligned)