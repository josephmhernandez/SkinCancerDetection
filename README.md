# SkinCancerDetection


Pi Errors.pdf - Common problems that were solved with transfering the project to the raspberry pi. 

SkinCancer.ipynb
Python notebook to load in the training, testing, and validation set from google drive and build the convnet model to classify 7 different types of skin lesions. 

camera_functions.py
Tests the usb webcam functionality to take pictures. 

create_dataset.py
Python script to format the images to the desired size. The goal was to find the largest images sizes that could be trained without causing a memory problem in google colab. 

skinCancerDetection.py
Uploaded to the raspberry pi to interact with the screen to display camera feed, take pictures, classify pictures, and to display the diagnosis. 
