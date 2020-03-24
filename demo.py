import cv2
import numpy as np
from Utils import Utils

utils = Utils(width=1920, height=1080)
utils.Init_model()
utils.Load_model_weights()
    
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FPS, 30)

detect_eye = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')

while True:

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    
    # Detect eye
    rects = detect_eye.detectMultiScale(frame, minNeighbors=20)
  
    if len(rects) > 0:
        
        # Get bounding box
        x, y, box_w, box_h = rects[0]
        
        # Crop the eye
        cropped_eye = frame[y:y + box_h,
                            x:x + box_w]
        
        # Display the eye
        cv2.imshow('eye', cropped_eye)

        # Get prediction of coordinates from the cropped eye
        x_pred, y_pred = utils.Predict_coordinates(cropped_eye)
 
        # Draw box and yellow dot
        utils.draw_box(x_pred,y_pred)
        
cap.release()
cv2.destroyAllWindows()