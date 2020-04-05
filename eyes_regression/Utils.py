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
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

class Utils:
    
    def __init__(self, width, height):
        
        # Screen size
        self.width = width 
        self.height = height
        
        # Inner box
        self.inner_w = 960
        self.inner_h = 600
        self.offset_w = int((self.width - self.inner_w)/2)
        self.offset_h = 50
        
        self.file_number = len(os.listdir('data/training_images/train')) + 1
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Eye detector
        self.detect_left_eye = cv2.CascadeClassifier('data/haarcascade_left_eye.xml')
        self.detect_right_eye = cv2.CascadeClassifier('data/haarcascade_right_eye.xml')
        self.minNeighbours = 120 # sensitivity --> lower = more sensitive
        
        # For Training
        self.image_size = 70
        self.train_path = 'data/training_images/train/'
        self.test_path = 'data/training_images/test/'
        self.model = None
        self.epochs= 150
        self.model_weights='data/model_weights.hdf5'
        
    #=========================================#
    #=========================================#
    #=========================================#
    
    def draw_dots(self):
    
        '''
        Function randomly draws a yellow dot on the screen for user
        to focus on.
        
        Output: Co-ordiantes (x,y) of the focal point
        '''
        img = self.draw_outline()
        
        # Print the number of files in the training folder
        num_of_files = str(len(os.listdir(self.train_path)))
        cv2.putText(img,
                text=num_of_files,
                fontFace=self.font,
                fontScale=1, # font size
                color=(255,255,255),
                thickness=2,
                org=(40,40),
                lineType=cv2.LINE_AA)

        # Generate a random coordinate
        x = int(random.random()*self.inner_w + self.offset_w)
        y = int(random.random()*self.inner_h + self.offset_h)
        random_point = (x,y)

        # Draw the circle at (x,y)
        cv2.circle(img, center=random_point, radius=5, thickness=-1, color=(4, 173, 59))
        cv2.circle(img, center=random_point, radius=20, thickness=2, color=(3, 237, 91))
        
        # Show the image
        cv2.imshow('display_dots', img)

        return random_point
    
    #=========================================#
    #=========================================#
    #=========================================#
    
    def draw_outline(self):
    

        # Start with a black background 
        img = np.zeros((self.height, self.width, 3))
        
        # Print bounding rectangle
        img = cv2.rectangle(img, pt1=(self.offset_w, self.offset_h), pt2=(self.offset_w + self.inner_w, self.offset_h+ self.inner_h),
                            color=(55,255,55),
                            thickness=3)
        
        return img
    
    #=========================================#
    #=========================================#
    #=========================================#
    
    def detect(self, image, random_point):
    
        '''
        Function to detect and save user's right eye using Haar Cascades 
        (1) Detect right eye
        (2) Draw bounding box
        (3) Crop eye
        (4) Save eye image in folder (self.train_path) with the corresponding coordinates.
        '''

        # Detect. Get (x,y,w,h) coordinates for right eye
        l_eye = self.detect_left_eye.detectMultiScale(image, minNeighbors=self.minNeighbours)
        r_eye = self.detect_right_eye.detectMultiScale(image, minNeighbors=self.minNeighbours)

        self.detect_and_save(l_eye, image, random_point)
            
        self.detect_and_save(r_eye, image, random_point)
        image = cv2.resize(image, (710,400))
        cv2.imshow('display', image)
        
    #=========================================#
    #=========================================#
    #=========================================#
    def detect_and_save(self, eye, image, random_point):
        
        if len(eye) > 0:
            
            x,y,w,h = eye[0]
        
            # Show the rectangle on the display
            cv2.rectangle(image, (x,y), (x+w, y+h), thickness=1, color=(0,255,0))
            
            # Crop the image
            crop = image[y:y+h, x:x+w]
     
            # Get coordinates
            rand_x, rand_y = str(random_point[0]), str(random_point[1])
            
            # Save 25% of images in test, 75% in train
            if random.randint(1,4) == 1:
                cv2.imwrite(self.test_path + rand_x +'_' + rand_y + '_'+ str(self.file_number) +'.tif', crop)
            else:
                cv2.imwrite(self.train_path + rand_x +'_' + rand_y + '_'+ str(self.file_number) +'.tif', crop)
            self.file_number += 1
         
    #=========================================#
    #=========================================#
    #=========================================#       
    
    def preprocess_data(self, data_type):
        
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
        
        # Select files with the chosen file_format

        files =  os.listdir(path)
        n_images = len(files)
        
        print('Processing data...' + str(n_images) + ' files found.')
        
        X = np.zeros((n_images, self.image_size, self.image_size, 3))
        Y = np.zeros((n_images,2),dtype=np.int16)

        for i, file_name in enumerate(files):
            print(file_name)
            img = cv2.imread(path + "/" + file_name)
            img = cv2.resize(img, (self.image_size, self.image_size))
            img = img.reshape(self.image_size, self.image_size,3 )
            img = (img-127.5) / 255
            X[i] = img
       
            # Get coordinates from the filename 
            file_name = file_name.split('_')
            x,y = int(file_name[0]), int(file_name[1])
            Y[i]= np.array([x,y])
    
        print(data_type,"|| X ",X.shape,"|| Y ", Y.shape)
        return X, Y    


    #=========================================#
    #=========================================#
    #=========================================#
    
    def init_model(self):
        
        '''
        Create and save model as class attribute.
        '''
    
        i = Input(shape=(self.image_size, self.image_size, 3))
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
        x = Dense(1024, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu')(x) 
        x = Dropout(0.2)(x)
        x = Dense(2)(x) #output is a dense of size 2 (x,y)

        self.model = Model(inputs=i, outputs=x)
        print(self.model.summary())
        
    #=========================================#
    #=========================================#
    #=========================================#            
        
    def train_model(self, X_train, Y_train, X_test, Y_test):
        
        '''
        Trains the model, and displays the train/val errors as a function of epochs.

        '''
        
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008, 
                                                         beta_1=0.9, 
                                                         beta_2=0.999, 
                                                         amsgrad=False),
                      loss='mean_squared_error',
                      metrics=['mean_squared_error'])
        
        checkpoint = ModelCheckpoint(self.model_weights, 
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
    
    def show_results(self):
        
        '''
        Plots a scatter plot of the actual (x,y) coordinates against 
        predicted (x,y) for the test set.
        '''
        
        def assign_color(x):
            if x <= 30:
                return '#000000'
            elif x <= 60:
                return '#140b78'
            elif x <= 90:
                return '#0b782f'
            elif x <= 120:
                return '#e0b814'
            elif x <= 150:
                return '#e07314'
            else: 
                return '#e01414'
        
        if self.model is None:
            return "\nNo model found. Please train the model first"
            
        else:
            results = []
            for file in os.listdir(self.test_path):
                img = cv2.imread(self.test_path + file)
                img = cv2.resize(img, (self.image_size, self.image_size))
                img = img.reshape(1, self.image_size, self.image_size,3)
                img = (img-127.5)/255.0
                pred = self.model.predict(img)[0]

                pred_x, pred_y = int(pred[0]), int(pred[1])
                coord_x = int(file.split('_')[0])
                coord_y = int(file.split('_')[1])
                error_x = pred_x-coord_x
                error_y = pred_y-coord_y
                err = np.sqrt(error_x**2 + error_y**2)
                results.append([file, coord_x, error_x, coord_y, error_y, err])

            df = pd.DataFrame(results, columns=["Filename", "actual_x", "error_x", "actual_y", "error_y", "total_error"])
            
            df['color'] = df['total_error'].apply(lambda x: assign_color(x))
            
            #Show results
            fig, ax = plt.subplots()
            ax.quiver(df['actual_x'], -df['actual_y'], df['error_x'], -df['error_y'], width=0.002)
            ax.scatter(df['actual_x'],-df['actual_y'], s=7, c=df['color'].tolist()),
            plt.show()

    #=========================================#
    #=========================================#
    #=========================================#                   
            
    def load_model_weights(self):
        
        '''
        Load model weights from eye_model.hdf5 file
        '''
        
        if self.model is not None:
            try:
                self.model.load_weights('data/model_weights.hdf5')
                print("\nWeights successfully loaded\n")
            except:
                print('''\nError loading weights into model. Check file path to the weights, 
                          or ensure the model is compatible with the saved weights\n''')  
        else:
            print("\nYou need to initialize the model first\n")
            

    #=========================================#
    #=========================================#
    #=========================================#      
    
    def predict_coordinates(self,cropped_eye):

        '''
        Function takes in the image of the cropped eye, and returns
        the predicted x and y coordinates as a tuple.
        '''
        if cropped_eye is not None:

            cropped_eye = cv2.resize(cropped_eye, (self.image_size, self.image_size))

            cropped_eye = cropped_eye.reshape(1, self.image_size, self.image_size,3)

            cropped_eye = (cropped_eye-127.5)/255

            pred = self.model.predict(cropped_eye)[0]

            x_pred, y_pred = pred[0], pred[1]

            return int(abs(x_pred)), int(abs(y_pred))
        
        return 0,0
    
    #=========================================#
    #=========================================#
    #=========================================# 
    
    def draw_box(self, x_pred, y_pred):
        
        '''
        Takes in predicted x and y coordinates and displays a dot.
        '''

        img = self.draw_outline()

        cv2.circle(img, center=(x_pred, y_pred), radius=10, thickness=-1, color=(0,255,0))

        cv2.imshow('display_dots', img)

    #=========================================#
    #=========================================#
    #=========================================# 
    
    def draw_buttons(self, x, y, w, h, x_pred, y_pred, img):
        
        if x_pred >= x1 and x_pred <= x2 and y_pred >= y1 and y_pred <= y2:
            
            cv2.rectangle(img, pt1=(x, y), pt2=(x+w, y+h), color=(255,255,255), thickness=-1)
            
        else:
            
            cv2.rectangle(img, pt1=(x, y), pt2=(x+w, y+h), color=(255,0,0), thickness=2)
            
        return img