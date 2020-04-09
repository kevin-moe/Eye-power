import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
import os
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, GlobalMaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

class Utils:
    
    def __init__(self):
        
        self.file_number = len(os.listdir('data/training_images/train')) + 1
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.previous = 0
        self.buttons = [(45,35,200,195), (350,35,505,195), (655,35,805,195), (960,35,1110,195),
                   (45,335,200,495), (350,335,505,495), (655,335,805,495), (960,335,1110,495),
                   (45,635,200,795), (350,635,505,795), (655,635,805,795), (960,635,1110,795)]
        
        # Eye detector
        self.detect_left_eye = cv2.CascadeClassifier('data/haarcascade_left_eye.xml')
        self.detect_right_eye = cv2.CascadeClassifier('data/haarcascade_right_eye.xml')
        self.minNeighbours = 120 # sensitivity --> lower = more sensitive
        
        # For Training
        self.image_size = 70
        self.train_path = 'data/training_images/train/'
        self.test_path = 'data/training_images/test/'
        self.model = None
        self.epochs= 20
        self.model_weights='data/model_weights.hdf5'
        self.classes = 12
        
        # For tracking
        self.target_tracker = [-1,-1,-2]
        self.preds = []

        
    #=========================================#
    #=========================================#
    #=========================================#
    
    def draw_frames(self, Y):
    
        '''
        Function randomly draws a yellow dot on the screen for user
        to focus on.
        
        Output: Co-ordiantes (x,y) of the focal point
        ''' 
        img = cv2.imread('data/background.jpg')
        
        # Print the number of files in the training folder
        num_of_train = str(len(os.listdir(self.train_path)))
        num_of_test = str(len(os.listdir(self.test_path)))
        
        cv2.putText(img,
                text= "Train: " + num_of_train + ", Test: " + num_of_test,
                fontFace=self.font,
                fontScale=0.5, # font size
                color=(255,255,255),
                thickness=1,
                org=(50,25),
                lineType=cv2.LINE_AA)
                
        for i, (a,b,c,d) in enumerate(self.buttons):
            
            if i==Y:
        
                color = (0,255,0)

                img = cv2.rectangle(img, 
                                  pt1=(a,b), 
                                  pt2=(c,d), 
                                  thickness=5, 
                                  color=color)
                break
            
        # Show the image
        cv2.imshow('display_dots', img)
    
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
        
        self.crop_and_save(l_eye, image, Y)
        self.crop_and_save(r_eye, image, Y)
        
        image = cv2.resize(image, (350,200))
        
        cv2.imshow('display', image)
        
    #=========================================#
    #=========================================#
    #=========================================#
    def crop_and_save(self, eye, image, Y):
        
        if len(eye) > 0:
            
            x,y,w,h = eye[0]
        
            # Crop the image
            crop = image[y:y+h, x:x+w]
            
            # Save 20% of images in test
            if random.randint(1,5) == 1:
                cv2.imwrite(self.test_path + str(Y) + '_'+ str(self.file_number) +'.tif', crop)
            else:
                cv2.imwrite(self.train_path + str(Y) +'_' + str(self.file_number) +'.tif', crop)
                
            self.file_number += 1
            
            # Show the rectangle on the display
            cv2.rectangle(image, (x,y), (x+w, y+h), thickness=3, color=(0,255,0))
         
    #=========================================#
    #=========================================#
    #=========================================#       
    
    
    def preprocess_data(self, data_type, file_format):
        
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
        files = []
        for file_name in os.listdir(path):
            files.append(file_name)
        
        n_images = len(files)
        
        print('Processing data...' + str(n_images) + ' files found.')
        
        X = np.zeros((n_images, self.image_size, self.image_size, 3))
        Y = np.zeros((n_images,self.classes), dtype=np.int16)

        for i, file_name in enumerate(files):

            img = cv2.imread(path + "/" + file_name)
            img = cv2.resize(img, (self.image_size, self.image_size))
            img = img.reshape(self.image_size, self.image_size, 3)
            img = (img-127.5) / 255.0
            X[i] = img
       
            # Get coordinates from the filename 
            file_name = file_name.split('_')
            k = int(file_name[0])
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
        x = Dense(self.classes, activation='softmax')(x) #output is a dense of size 2 (x,y)

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
    
    def get_probability(self, cropped_eye):

        '''
        Function takes in the image of the cropped eye, and returns
        the predicted x and y coordinates as a tuple.
        '''
        if cropped_eye is not None:
            
            cropped_eye = cv2.resize(cropped_eye, (self.image_size, self.image_size))

            cropped_eye = cropped_eye.reshape(1, self.image_size, self.image_size,3)

            cropped_eye = (cropped_eye-127.5)/255.0

            probabilities = self.model.predict(cropped_eye)[0]
            
            return probabilities
        
        return np.zeros(self.classes)

    
    #=========================================#
    #=========================================#
    #=========================================# 
    
    def draw_box(self, pred):
        
        '''
        Takes in predicted x and y coordinates and displays a dot.
        '''
        
        #self.preds.append(pred)
        #seq = ' '.join([str(x) for x in self.preds])
        
        img = cv2.imread('data/background.jpg')
#         img = cv2.putText(img,
#                 text= seq,
#                 fontFace=self.font,
#                 fontScale=1, # font size
#                 color=(255,255,255),
#                 thickness=1,
#                 org=(50,50),
#                 lineType=cv2.LINE_AA)
        
        for i, (a,b,c,d) in enumerate(self.buttons):

            if i==pred:

                img = cv2.rectangle(img, pt1=(a,b), pt2=(c,d), thickness=5, color=(0,255,0))

        cv2.imshow('display_dots', img)


    
