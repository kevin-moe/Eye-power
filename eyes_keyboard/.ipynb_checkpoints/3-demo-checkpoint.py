import cv2
import numpy as np
import time
from Utils import Utils

##
utils = Utils()
utils.init_model()
utils.load_model_weights()
    
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
l_eye = cv2.CascadeClassifier('data/haarcascade_left_eye.xml')
r_eye = cv2.CascadeClassifier('data/haarcascade_right_eye.xml')
window = 4

# ============================================#
# ============================================#

def crop_eye(eye, image):
    
    cropped_eye = None
    
    if len(eye) > 0:
        
        # Get bounding box
        x, y, box_w, box_h = eye[0]
        
        # Crop the eye
        cropped_eye = image[y:y + box_h,
                            x:x + box_w]
        
    return cropped_eye

# ============================================#
# ============================================#

target_tracker = [-2,-1,-1]

while True:

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    # Detect eyes
    left = l_eye.detectMultiScale(frame, minNeighbors=120)
    right = r_eye.detectMultiScale(frame, minNeighbors=120)
    
    # Crop images
    left_img = crop_eye(left, frame)
    right_img = crop_eye(right, frame)
    
  
    # Get prediction of coordinates from the cropped eye
    if left_img is not None and right_img is not None:
        
        cv2.imshow('frame', left_img)
        
        left_prob = utils.get_probability(left_img)
        right_prob = utils.get_probability(right_img)
        
        prob = (left_prob + right_prob) /2
        
        pred = np.argmax(prob)
        
        target_tracker.append(pred)
    
        # if the latest entry is the same as the last k number of entries
        if target_tracker[-1] == int(sum(target_tracker[-window:])/window):

            utils.draw_box(pred)
                  
            # Reset target tracker
            target_tracker = [-2,-1,-1]
            
    else:
        
        print('Cannot detect left and right eye. Please lower detection sensitivity.')
    
    
cap.release()
cv2.destroyAllWindows()