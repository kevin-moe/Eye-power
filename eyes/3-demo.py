import cv2
import numpy as np
from Utils import Utils
##
utils = Utils(width=1920, height=1080)
utils.init_model()
utils.load_model_weights()
    
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
l_eye = cv2.CascadeClassifier('data/haarcascade_left_eye.xml')
r_eye = cv2.CascadeClassifier('data/haarcascade_right_eye.xml')

x_pred_R, y_pred_R, x_pred_L, y_pred_L = 0,0,0,0

def crop_eye(eye, image):
    
    cropped_eye = None
    
    if len(eye) > 0:
        
        # Get bounding box
        x, y, box_w, box_h = eye[0]
        
        # Crop the eye
        cropped_eye = image[y:y + box_h,
                            x:x + box_w]
        
    return cropped_eye


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
    if left_img is not None:
        x_pred_L, y_pred_L = utils.predict_coordinates(left_img)
    if right_img is not None:
        x_pred_R, y_pred_R = utils.predict_coordinates(right_img)
     
    x_pred = int((x_pred_L + x_pred_R) /2)
    y_pred = int((y_pred_L + y_pred_R) /2)
    
    # Draw box and yellow dot
    utils.draw_box(x_pred,y_pred)
        
cap.release()
cv2.destroyAllWindows()