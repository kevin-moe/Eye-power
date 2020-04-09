from Utils import Utils
import os
import numpy as np
import cv2

# Set video parameters #
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

# Import utility functions #
utils = Utils()

# Get train sequence
train_sequence = []
class_size=10

for i in range(38):
    temp = [i] * class_size
    train_sequence += temp
        
np.random.shuffle(train_sequence)

# ===============================================#
#============ Run the capture loop ==============#
# ===============================================#
for i, Y in enumerate(train_sequence):
    
    print(str(len(train_sequence)-i)+ " to go. Class: " + str(Y))
    
    # Draw the grid
    utils.draw_frames(Y)
    
    # Press 'c' to capture
    while True:
        if cv2.waitKey(1) & 0xFF == ord('c'):
            break
    
    # Capture the camera's image
    ret, frame = cap.read()
    
    # flip image so that it becomes a mirror reflection
    frame = cv2.flip(frame, 1) 

    # Detect user's eye and capture the image
    utils.detect(frame, Y)
        
cap.release()
cv2.destroyAllWindows()