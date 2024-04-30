import cv2
import os

def flip_images():
    DATA_DIR = "./data"
    
    for g_id in range(3):  
        subdirectory = os.path.join(DATA_DIR, str(g_id))
        
        for i in range(100):
            original_path = os.path.join(subdirectory, str(i+1) + ".jpg")
            new_path = os.path.join(subdirectory, str(i+1 + 99) + ".jpg")
            
            print(original_path)
            
            img = cv2.imread(original_path, 0)
            
            if img is not None:
                img = cv2.flip(img, 1)
                cv2.imwrite(new_path, img)
    

flip_images()
