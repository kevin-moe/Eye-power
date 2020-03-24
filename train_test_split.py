import os
import random
import shutil
'''
Script to move a random 10% of the files from the 
training folder to the test folder. 
'''
ratio = 0.1

# path to training folder
train_path = 'eye/train/'
test_path = 'eye/test/'

files = os.listdir(train_path)

# Number of training files
number_of_files = len(files)

# Select the files to transfer
files = random.sample(files, k=int(ratio*number_of_files))

for file in files:
    shutil.move(train_path + file, test_path + file)
    
print(str(int(number_of_files*ratio)) + " transferred from train to test.")