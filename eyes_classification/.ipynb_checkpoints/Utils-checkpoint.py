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
        
        self.file_number = len(os.listdir('data/training_images')) + 1
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
        l = 320
        b = 200
            
        img = np.zeros((self.height,self.width, 3))
        
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
        
        buttons = [(480,50,800,250),(800,50,1120,250),(1120,50,1440,250),
                   (480,250,800,450),(800,250,1120,450),(1120,250,1440,450),
                   (480,450,800,650),(800,450,1120,650),(1120,450,1440,650)]
        
        Y = random.randint(0,len(buttons)-1)
        print(Y)
        for i, (a,b,c,d) in enumerate(buttons):
            
            if i==Y:
                t=-1
            else:
                t=2
                
            img = cv2.rectangle(img, 
                          pt1=(a,b), 
                          pt2=(c,d), 
                          thickness=t, 
                          color=(255,255,255))
            
        # Show the image
        cv2.imshow('display_dots', img)
        
        return Y
    
    #=========================================#
    #=========================================#
    #=========================================#
    
    def detect(self, image, Y):
    
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

        self.detect_and_save(l_eye, image, Y)
            
        self.detect_and_save(r_eye, image, Y)
        
        image = cv2.resize(image, (710,400))
        
        cv2.imshow('display', image)
        
    #=========================================#
    #=========================================#
    #=========================================#
    def detect_and_save(self, eye, image, Y):
        
        if len(eye) > 0:
            
            x,y,w,h = eye[0]
        
            # Show the rectangle on the display
            cv2.rectangle(image, (x,y), (x+w, y+h), thickness=1, color=(0,255,0))
            
            # Crop the image
            crop = image[y:y+h, x:x+w]
            
            # Save 25% of images in test, 75% in train
            if random.randint(1,4) == 1:
                cv2.imwrite(self.test_path + str(Y) + '_'+ str(self.file_number) +'.jpg', crop)
            else:
                cv2.imwrite(self.train_path + str(Y) +'_' + str(self.file_number) +'.jpg', crop)
                
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
            
        n_images = len(os.listdir(path))
        
        print('Processing data...')
        
        X = np.zeros((n_images, self.image_size, self.image_size, 1))
        Y = np.zeros((n_images,9),dtype=np.int16)

        for i, file in enumerate(os.listdir(path)):

            img = cv2.imread(path + "/" + file)
            img = img[:,:,0] # take the first layer of the image
            img = cv2.resize(img, (self.image_size, self.image_size))
            img = img.reshape(self.image_size, self.image_size,1 )
            img = img / 255.0
            X[i] = img
       
            # Get coordinates from the filename (e.g. "12_(120,340).jpg" --> (120,340))
            file = file.split('_')
            k = int(file[0])
            Y[i,k]= 1
    
        print(data_type,"|| X ",X.shape,"|| Y ", Y.shape)
        return X, Y

    #=========================================#
    #=========================================#
    #=========================================#
    
    def init_model(self):
        
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
        x = Dense(1024, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu')(x) 
        x = Dropout(0.2)(x)
        x = Dense(9, activation='softmax')(x) #output is a dense of size 2 (x,y)

        self.model = Model(inputs=i, outputs=x)
        print(self.model.summary())
        
    #=========================================#
    #=========================================#
    #=========================================#            
        
    def train_model(self, X_train, Y_train, X_test, Y_test):
        
        '''
        Trains the model, and displays the train/val errors as a function of epochs.

        '''
        
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, 
                                                         beta_1=0.9, 
                                                         beta_2=0.999, 
                                                         amsgrad=False),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        checkpoint = ModelCheckpoint(self.model_weights, 
                                     monitor='val_accuracy', 
                                     verbose=1, 
                                     save_best_only=True, 
                                     mode='max')
        
        callbacks_list = [checkpoint]
    
        R = self.model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=self.epochs, callbacks=callbacks_list)   
            
        # Show training history
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_ylabel('Acc')
        ax1.set_xlabel('Epochs')
        ax1.plot(R.history['val_accuracy'], label ='val_acc')
        ax1.plot(R.history['accuracy'], label='acc')
        plt.legend()
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
            
            cropped_eye = cropped_eye[:,:,1]

            cropped_eye = cv2.resize(cropped_eye, (self.image_size, self.image_size))

            cropped_eye = cropped_eye.reshape(1, self.image_size, self.image_size,1)

            cropped_eye = cropped_eye/255.0

            pred = self.model.predict(cropped_eye)[0]
            
            box = np.argmax(pred)
            print('HERE: ', box)
            prob = pred[box]
            
            return box, prob
        
        return 0,0
    
    #=========================================#
    #=========================================#
    #=========================================# 
    
    def draw_box(self, pred):
        
        '''
        Takes in predicted x and y coordinates and displays a dot.
        '''
        img = np.zeros((self.height,self.width, 3))
        
        buttons = [(480,50,800,250),(800,50,1120,250),(1120,50,1440,250),
                   (480,250,800,450),(800,250,1120,450),(1120,250,1440,450),
                   (480,450,800,650),(800,450,1120,650),(1120,450,1440,650)]
   
        for i, (a,b,c,d) in enumerate(buttons):
            
            if i==pred:
                t=-1
            else:
                t=2
                
            img = cv2.rectangle(img, 
                          pt1=(a,b), 
                          pt2=(c,d), 
                          thickness=t, 
                          color=(255,255,255))

        cv2.imshow('display_dots', img)


    
