from ast import Global
from imaplib import Int2AP
import threading
from tkinter.tix import InputOnly
import serial

import time
import cv2 
import time
class cameraFilm:
    def film(self, img):
        global frame
        while True:
            ignore, frame = cam.read()
            cv2.imshow('camera', frame)
            img = frame
            return img
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
    def __init__(self):
        t = threading.Thread(target=self.film)
        t.start()
    def write_read(x):
        arduino.write(bytes(x, 'utf-8'))
        time.sleep(0.05)
        data = arduino.readline()
        return data        

width = 1080
height = 720
cam=cv2.VideoCapture(1, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))


    
arduino =  serial.Serial(port='COM2', baudrate=115200, timeout=.1)


intInput = ""


while True:
    

    if (intInput == "on Lego"):
        arduino.write(intInput.encode())
    if (intInput == "off Lego"):
        arduino.write(intInput.encode())
    if (intInput == "on Stepper"):
        arduino.write(intInput.encode())
    if (intInput == "off Stepper"):
        arduino.write(intInput.encode())
    val = arduino.readline()
    print(val)
    

    time.sleep(0.05)
    intInput = input()
    
        

        

    #on
    # 
    # serialcomm.write(i.encode())

   
    
    #  print(serialcomm.readline().decode('ascii'))
cam.release()
serialcomm.close()