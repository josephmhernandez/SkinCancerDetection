import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
def taking_picture():
    #Starts the camera to take a picture. Waits for button to be pressed to take a picture. 
    #Saves the picture. 
    #Return image name. 
    cam = cv2.VideoCapture(1) 
    cv2.namedWindow("Skin Cancer Detection")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        cv2.imshow("test", frame)
        if not ret:
            break
        k = cv2.waitKey(1)


        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break


        elif k%256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written".format(img_name))
            break
            #Exit on picture taken. 
            break
            img_counter += 1

    cam.release()

    cv2.destroyAllWindows()
    return img_name

def process_image(path):
    _img_arr = cv2.imread(path, cv2.COLOR_BGR2RGB)
    _img_arr = cv2.cvtColor(_img_arr, cv2.COLOR_BGR2RGB)
    new_arr = cv2.resize(_img_arr, (256, 192))

    new_arr = np.array(new_arr)
    return new_arr



#Testing. 
j = process_image(taking_picture())


plt.imshow(j)
plt.show()
