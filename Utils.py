import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
import os
import shutil

import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, GlobalMaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

class Utils:
    
    def __init__(self, width, height):
        
        self.width = width
        self.height = height
        self.file_number = len(os.listdir('eye/train')) + 1
        
        # Eye detector
        self.detect_eye = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
        self.minNeighbours = 20 # sensitivity --> lower = more sensitive
        
        # For Training
        self.image_size = 72
        self.train_path = 'eye/train'
        self.test_path = 'eye/test'
        self.model = None
        self.epochs= 80
        
    #=========================================#
    #=========================================#
    #=========================================#
    
    def draw_dots(self):
    
        '''
        Function randomly draws a yellow dot on the screen for user
        to focus on.
        
        Output: Co-ordiantes (x,y) of the focal point
        '''

        # Start with a black background
        img = np.zeros((self.height, self.width, 3))

        # Generate a random coordinate
        x = int(random.random()*self.width)
        y = int(random.random()*self.height)
        random_point = (x,y)

        # Draw the circle at (x,y)
        cv2.circle(img, center=random_point, radius=20, thickness=-1, color=(0,255,55))

        # Show the image
        cv2.imshow('display_dots', img)

        return random_point
    
    #=========================================#
    #=========================================#
    #=========================================#
    
    def detect(self, image, random_point):
    
        '''
        Function to detect and save user's right eye using Haar Cascades 
        (1) Detect right eye
        (2) Draw bounding box
        (3) Crop eye
        (4) Save eye image in folder ("eye/train") with the corresponding coordinates.
        '''

        # Detect. Get (x,y,w,h) coordinates for right eye
        rects = self.detect_eye.detectMultiScale(image, minNeighbors=self.minNeighbours)

        if len(rects) > 0:
            x,y,w,h = rects[0]

            # Show the rectangle on the display
            cv2.rectangle(image, (x,y), (x+w, y+h), thickness=2, color=(255,0,0))

            # Crop the image
            crop = image[y:y+h, x:x+w]
     
            x = str(random_point[0])
            y = str(random_point[1])

            # Save the cropped image
            cv2.imwrite('eye/train/' + x +'_' + y + '_'+ str(self.file_number) +'.jpg', crop)
            self.file_number += 1

        cv2.imshow('display', image)
        
    #=========================================#
    #=========================================#
    #=========================================#

    def Preprocess_data(self, data_type):
        
        '''
        Function creates image arrays (X) and labels (Y)
        
        Input: String indicating whether to preprocess 'train' or 'test' data.
        
        Output: X, Y 
            where X.shape=(n, width, height, depth), 
                  Y.shape= (n, 2)
        '''
        
        if data_type == 'train':
            path = self.train_path
        elif data_type == 'test':
            path = self.test_path
        else:
            return "Invalid path"
            
        n_images = len(os.listdir(path))

        X = np.zeros((n_images, self.image_size, self.image_size, 1))
        Y = np.zeros((n_images,2),dtype=np.int16)

        for i, file in enumerate(os.listdir(path)):

            img = cv2.imread(path + "/" + file)
            img = img[:,:,0] # take the first layer of the image
            img = cv2.resize(img, (self.image_size, self.image_size))
            img = img.reshape(self.image_size, self.image_size,1 )
            img = img / 255.0
            X[i] = img
            
            # Get coordinates from the filename (e.g. "12_(120,340).jpg" --> (120,340))
            file = file.split('_')
            n1 = int(file[0])
            n2 = int(file[1])
            Y[i] = np.array([n1,n2])
            
        print(data_type,"|| X ",X.shape,"|| Y ", Y.shape)

        return X, Y

    #=========================================#
    #=========================================#
    #=========================================#
    
    def Init_model(self):
        
        '''
        Create and save model as class attribute.
        '''
    
        i = Input(shape=(self.image_size, self.image_size, 1))
        x = Conv2D(32, (3,3), activation='relu')(i)
        x = MaxPooling2D(2,2)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Conv2D(64, (3,3), activation='relu')(x)
        x = MaxPooling2D(2,2)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Conv2D(128, (3,3), activation='relu')(x)
        x = MaxPooling2D(2,2)(x)
        x = BatchNormalization()(x)
        x = GlobalMaxPooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(2)(x) #output is a dense of size 2 (x,y)

        self.model = Model(inputs=i, outputs=x)
        print(self.model.summary())
        
    #=========================================#
    #=========================================#
    #=========================================#            
        
    def Train_model(self, X_train, Y_train, X_test, Y_test):
        
        '''
        Trains the model, and displays the train/val errors as a function of epochs.

        '''
        
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, 
                                                         beta_1=0.9, 
                                                         beta_2=0.999, 
                                                         amsgrad=False),
                      loss='mean_squared_error',
                      metrics=['mean_squared_error'])
        
        checkpoint = ModelCheckpoint("eye_model.hdf5", 
                                     monitor='val_mean_squared_error', 
                                     verbose=1, 
                                     save_best_only=True, 
                                     mode='min')
        
        callbacks_list = [checkpoint]
    
        R = self.model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=self.epochs, callbacks=callbacks_list)          
            
        # Show training history
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_ylabel('MSE')
        ax1.set_xlabel('Epochs')
        ax1.plot(R.history['val_loss'], label ='val_mean_sq_error')
        ax1.plot(R.history['loss'], label='mean_sq_error')
        plt.legend()
        plt.show()

    #=========================================#
    #=========================================#
    #=========================================#
    
    def Show_results(self):
        
        '''
        Plots a scatter plot of the actual (x,y) coordinates against 
        predicted (x,y) for the test set.
        '''
        
        if self.model is None:
            return "\nNo model found. Please train the model first"
            
        else:
            results = []
            for file in os.listdir('eye/test'):
                img = cv2.imread('eye/test/' + file)
                img = img[:,:,0]
                img = cv2.resize(img, (self.image_size, self.image_size))
                img = img.reshape(1, self.image_size, self.image_size,1)
                img = img/255.0
                pred = self.model.predict(img)[0]

                pred_x, pred_y = int(pred[0]), int(pred[1])
                coord_x = int(file.split('_')[0])
                coord_y = int(file.split('_')[1])
                dist = np.sqrt((coord_x-pred_x)**2 + (coord_y-pred_y)**2)
                results.append([file, coord_x, pred_x, coord_y, pred_y, dist])

            df = pd.DataFrame(results, columns=["Filename", "actual_x", "pred_x", "actual_y", "pred_y","Distance"])
            df.to_csv("results.csv")
            
            #Show results
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.set_ylabel('Predicted')
            ax1.set_xlabel('Actual')
            ax1.scatter(df['actual_x'], df['pred_x'], label='X coords')
            ax1.scatter(df['actual_y'], df['pred_y'], label='Y coords')
            plt.legend()
            plt.show()

    #=========================================#
    #=========================================#
    #=========================================#                   
            
    def Load_model_weights(self):
        
        '''
        Load model weights from eye_model.hdf5 file
        '''
        
        if self.model is not None:
            try:
                self.model.load_weights('eye_model.hdf5')
                print("\nWeights successfully loaded\n")
            except:
                print('''\nError loading weights into model. Check file path to the weights, 
                          or ensure the model is compatible with the saved weights\n''')  
        else:
            print("\nYou need to initialize the model first\n")
            

    #=========================================#
    #=========================================#
    #=========================================#      
    
    def Predict_coordinates(self, cropped_eye):

        '''
        Function takes in the image of the cropped eye, and returns
        the predicted x and y coordinates as a tuple.
        '''
        cropped_eye = cropped_eye[:,:,0]
            
        cropped_eye = cv2.resize(cropped_eye, (self.image_size, self.image_size))
        
        cropped_eye = cropped_eye.reshape(1, self.image_size, self.image_size,1)
        
        cropped_eye = cropped_eye/255.0

        pred = self.model.predict(cropped_eye)[0]
        
        x_pred, y_pred = pred[0], pred[1]

        return int(abs(x_pred)), int(abs(y_pred))
    
        
    def draw_box(self, x_pred, y_pred):
        
        '''
        Takes in predicted x and y coordinates and displays a dot.
        '''

        img = np.zeros((self.height, self.width, 3))

        cv2.circle(img, center=(x_pred, y_pred), radius=10, thickness=-1, color=(0,255,0))

        cv2.imshow('display_dots', img)