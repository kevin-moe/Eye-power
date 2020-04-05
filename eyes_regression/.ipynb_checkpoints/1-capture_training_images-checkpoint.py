from Utils import Utils
import os
import cv2
##
# Set video parameters #
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

# Import utility functions #
utils = Utils(width=1920, height=1080)

# ===============================================#
#============ Run the capture loop ==============#
# ===============================================#
while True:
    
    # Draw the grid
    dot_coordinates = utils.draw_dots()
    
    # Press 'q' to quit
    while True:
        if cv2.waitKey(1) & 0xFF == ord('c'):
            break
    
    # Capture the camera's image
    ret, frame = cap.read()
    
    # flip image so that it becomes a mirror reflection
    frame = cv2.flip(frame, 1) 

    # Detect user's eye and capture the image
    utils.detect(frame, dot_coordinates)
        
cap.release()
cv2.destroyAllWindows()