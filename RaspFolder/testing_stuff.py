#Script to run the SkinCancerDetection Prototype.
#General
import numpy as np

#Model
import keras
from keras.models import load_model
#Camera
import cv2

import time


#Process flow. 
import threading
import RPi.GPIO as IO
import Adafruit_MCP4725 as MC

def Button_On_Interrupt(pin):
    buttonEventPushed.set()
    #Display camera feed on screen. 

    pass

def Button_Picture_Interrupt():
    #Take picture. 
    #Feed picture through model. 
    #Display model feedback. 
    #Delete picture. 
    pass

def Speech_Recognition_Interrupt():
    #When picture taking is enabled. Use Webcam microphone to listen to to speech. 
    #If correct words than take picture. 
    pass

def Start_60_Seonnd_Til_Sleep():
    #Start a 60 second counter to shut off if no action has been done in the past 60 seconds. 
    #   Count should restart every time a button is pushed. 

    pass


def Take_Picture():
    #Takes and saves picture. Returns the image name. 
    pass


def Display_Webcam():
    #Display live feed from the webcam. 
    #Stop display once picture has been taken. 
    pass


def Display_FeedBack():
    #After prediction. Display information about the predicted skin leision. 
    pass

def Preprocess_Image(filename):
    #Preprocess and image to correct format for the model to consume. 
    _img_arr = cv2.imread(filename, cv2.COLOR_BGR2RGB)
    _img_arr = cv2.cvtColor(_img_arr, cv2.COLOR_BGR2RGB)
    new_arr = cv2.resize(_img_arr, (256, 192))

    new_arr = np.array(new_arr)
    print(new_arr.shape)

    return new_arr

def Generate_Model_Output(model, filename):
    #Parameters: image name, model. 
    input_data = Preprocess_Image(filename)

    prediction = model.predict(x = input_data.reshape(1, 192, 256, 3))

    pred_arr = prediction[0]
    max_index = np.argmax(pred_arr)
    print(max_index)

    cat_data = {
        'df' : 1,
        'nv' : 2,   
        'mel' : 3,  
        'bkl' : 4,  
        'bcc' : 5,  
        'akiec' : 6,    
        'vasc' : 0
    }
    for df in cat_data: 
        if cat_data[df] == max_index:
            print(df)
            break

    #Return the type of skin leision. 
    return None




if __name__ == '__main__':
    #Test image. 
    filename = 'ISIC_0024306.png'

    print("Starting.")


    #Load model 
    model = load_model('3sc35-084.hdf5')
    Generate_Model_Output(model, filename)
    print("Model loaded")


    #Button Push
    print('rasp button set up')
    buttonPin = 18
    IO.setmode(IO.BCM)
    IO.setwarnings(False)    
    IO.setup(buttonPin, IO.IN,pull_up_down=IO.PUD_UP)
    IO.add_event_detect(buttonPin, IO.FALLING, bouncetime=200)
    IO.add_event_callback(buttonPin, Button_On_Interrupt)
    buttonEventPushed = threading.Event()


    #Time out vars. 
    delay = 30
    now = time.time()

    while True:
        #Wait for button to be clicked to turn on the device. 
        while not buttonEventPushed.wait():
            print('Press Button')

        #While not time out. (60 second time out)
        while True:
            if time.time() > now + delay:
                break

            


        print("Timed out")
        #Turn off display
        #


            #thread to display camera output. 

    #Flow: 
    # button_pin = 23 #Enter button pin. 

    # thread_button_on = threading.Thread(target=Button_On_Interrupt)

    # thread_button_on.start()

    # event_button_on.wait()
    #Turn on display.
    #Do until the 60 second delay is hit.  
    
    #Start 60 second count when something is pushed.  
    #Wait for button to be pushed for picture
    #Wait for voice activation to be done. 
    #Wait for 60 seconds to be reached to make device sleep. (Turn off display)

    #When Picture triggered. 
        #Display picture on screen with 'loading ...' label
        #Display the classification with information. 
        #Wait for the 60 second delay
        #Wait for button to be pushed 

    #On 60 secod delay: Sleep and go back to begining. 

    #On button pushed back to "#Do until the 60 second delay is hit."


    print('here.')