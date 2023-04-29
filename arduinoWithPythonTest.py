from asyncio.windows_events import NULL
from inspect import BoundArguments
import time
import cv2
from pymata4 import pymata4
drive = [0]

spr = 200
degrees = spr/360
degrees2drive = degrees * 450 *2
print(degrees2drive)
def contect2Ardunio(board):

    while True:
        try:
            board = pymata4.Pymata4()
            
        except RuntimeError:
            cv2.waitKey(1000)
            print("retrying")
            continue
        else:
            break
        
    return board
board = NULL
board = contect2Ardunio(board=board)
board.set_pin_mode_digital_output(4) #Drive Pin
board.set_pin_mode_digital_output(3)
board.set_pin_mode_digital_output(6)
board.set_pin_mode_digital_output(7)
board.set_pin_mode_pwm_output(10)
 #Step Pin


 #Step Pin
board.digital_write(4, 1)
x=0
start = time.time()
while True:
    
    
    while x <= degrees2drive:
    
     #DirPin High

        board.digital_pin_write(3, 1)
        time.sleep(0.00035)
        board.digital_pin_write(3, 0)
    
        x=x+1
    break

end = time.time()
print(end-start)


 
    
    
