import cv2
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
            print("{} written!".format(img_name))

            #Exit on picture taken. 
            # return img_name
            img_counter += 1

    cam.release()

    cv2.destroyAllWindows()




taking_picture()