import numpy as np

def overlay_image(img, img_overlay):
    x, y, _ = img.shape
    a, b, _ = img_overlay.shape
    
    out = img
    out[0:a, 0:b] = img_overlay

    return out

def landmark_vector_dir(lm1, lm2):
    vec = np.array([lm1.x - lm2.x, lm1.y - lm2.y, lm1.z - lm2.z]) 
    return vec / np.linalg.norm(vec)