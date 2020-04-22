#Script to run the SkinCancerDetection Prototype.
#General
import numpy as np
import os

#Model
from tensorflow.keras.models import load_model  

#Camera
import cv2
import matplotlib.pyplot as plt

import time


#Process flow. 
import threading
import RPi.GPIO as IO

def Load_Model(model_name):
    global My_Model
    My_Model = load_model(model_name)

def Button_On_Interrupt(pin):
    if(Taking_Picture == True):
        global Picture_Taken
        Picture_Taken = True
        
    #Inable more button pushes. 
    buttonEventPushed.set()


def Display_FeedBack(diagnosis):
    #After prediction. Display information about the predicted skin leision. 
    print("Diagnosis: ", diagnosis)
    

def Preprocess_Image(filename):
    #Preprocess and image to correct format for the model to consume. 
    _img_arr = cv2.imread(filename, cv2.COLOR_BGR2RGB)
    _img_arr = cv2.cvtColor(_img_arr, cv2.COLOR_BGR2RGB)
    new_arr = cv2.resize(_img_arr, (256, 192))

    new_arr = np.array(new_arr)
    return new_arr

def Generate_Model_Output(model, filename):
    #Parameters: model, image name. 
    input_data = Preprocess_Image(filename)

    prediction = model.predict(x = input_data.reshape(1, 192, 256, 3))

    pred_arr = prediction[0]
    print(pred_arr)
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
    rtn_df = None
    for df in cat_data: 
        if cat_data[df] == max_index:
            rtn_df = df
            break
    
    #Return the type of skin leision. 
    return rtn_df

#Global variables. 
TimeOut = False
Taking_Picture = False
Picture_Taken = False

if __name__ == '__main__':
    #Test image. Debug.
    filename = 'ISIC_0024306.png'

    print("Starting.")
    print("Model load...")   
    mod_name = '3sc35-084.hdf5'
    My_Model = None
    #loader_thread = threading.Thread(name = 'loader_thread', 
    #    target=Load_Model, args=(mod_name,))
    #loader_thread.start()
    
    My_Model = load_model('3sc35-084.hdf5')

    #GPIO Button Set up. 
    print('rasp button set up')
    buttonPin = 21
    IO.setmode(IO.BCM)
    IO.setwarnings(False)    
    IO.setup(buttonPin, IO.IN,pull_up_down=IO.PUD_UP)
    IO.add_event_detect(buttonPin, IO.FALLING, bouncetime=200)
    IO.add_event_callback(buttonPin, Button_On_Interrupt)
    buttonEventPushed = threading.Event()

    #Time out vars. 
    delay = 100
    img_counter = None
    
    print('Start loop')
    while True:
        #Wait for button to be clicked to turn on the device. 
        print('Waiting for button to turn on device.....')
        while not buttonEventPushed.wait():
            print('Press Button')

        img_count = 0 
        now = time.time()
        TimeOut = False
        GO_ON = False
        
        cam = cv2.VideoCapture(0) 
        cv2.namedWindow("Push Button To Take Picture")
        
        #Reset button to take picture
        Taking_Picture = True
        buttonEventPushed = threading.Event()
        img_name = None
        
        while True:
            #Turn on display.
            ret, frame = cam.read()
            cv2.imshow("test", frame)   
            if not ret:
                break
                
            k = cv2.waitKey(1)

            if(time.time() > now + delay):
                TimeOut = True
            
            if(TimeOut):
                break
            
            if(Picture_Taken):
                Picture_Taken = False
                GO_ON = False
                img_count += 1 
                img_name = "image_{}.png".format(img_count)
                cv2.imwrite(img_name, frame)
                
                #Waiting for model to load.
                #print('waiting for model to load')
                #loader_thread.join()
                #print('Model has been loaded')
                #Picture has been taken. Now Process the picture. 
                
                diagnosis = Generate_Model_Output(My_Model, img_name)
                
                #TODO take in frame maybe.  
                Display_FeedBack(diagnosis)
                
                #Reset timeout. 
                now = time.time()
                TimeOut = False
                
                break
        
        

        print("Timed out\n\n")
        cam.release()
        cv2.destroyAllWindows()
        #Reset Button. 
        Taking_Picture = False
        buttonEventPushed = threading.Event()
        
        
        
        
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
