import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import random
import shutil
import json
from Utils import Utils
utils = Utils(width=1920, height=1080)

def show_distribution(data):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('Y-coord')
    ax1.set_xlabel('X-coord')
    ax1.scatter(data[:,0],data[:,1])
    plt.legend()
    plt.show()

if __name__ == '__main__':
    
    utils.init_model()

    # Prepare training data
    X_train, Y_train = utils.preprocess_data('train')
    
    X_test, Y_test = utils.preprocess_data('test')
    
    # Visualize the histogram of the x and y coordinates
    # If you have enough data points, they should fill the whole screen
    show_distribution(Y_train)
    
    # Train the model
    utils.train_model(X_train, Y_train, X_test, Y_test)
    
    # Show results
    utils.show_results()
