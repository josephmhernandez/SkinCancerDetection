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



To run on raspberry pi 3b: 
-Expand File System & make more room by deleting applications: 
sudo raspi-config
(Go to advanced options and expand file system)
sudo reboot
sudo apt-get purge wolfram-engine
sudo apt-get purge libreoffice*
sudo apt-get clean
sudo apt-get autoremove

-Make sure machine is updated:
sudo apt-get update && sudo apt-get upgrade

-Install Dependencies: 
sudo apt-get install libatlas-base-dev -y
sudo apt-get install libhdf5-dev -y
sudo apt-get install libhdf5-serial-dev -y
sudo apt-get install libjasper-dev -y
sudo apt-get install libqtgui4 -y
sudo apt-get install libqt4-test -y

-Install Python Packages:
python3 -m pip install --user virtualenv

-Inside Virtual environment: 
pip install opencv-python==4.2.0.34
pip install keras==2.3.1
pip install tensorflow==1.14.0
pip install numpy==1.18.3

-When running program run with this command:
LD_PRELOAD=/usr/lib/arm-linux-gnueabihf/libatomic.so.1 python3 skinCancerDetection.py

**Program will not work correctly if you don't have a usb capable camera plugged in. 
