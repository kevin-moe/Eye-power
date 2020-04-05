import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import random
import shutil
import json
from Utils import Utils
utils = Utils()

if __name__ == '__main__':
    
    utils.init_model()

    # Prepare training data
    X_train, Y_train = utils.preprocess_data('train', file_format='tif')
    
    X_test, Y_test = utils.preprocess_data('test', file_format='tif')
    
    # Train the model
    utils.train_model(X_train, Y_train, X_test, Y_test)
    
