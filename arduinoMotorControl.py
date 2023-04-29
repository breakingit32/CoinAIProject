from asyncio.windows_events import NULL
from pymata4 import pymata4
import cv2
import time

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

def motorOn(board, speed, speedPin, pin1, pin2, dir):
    board.pwm_write(speedPin, speed)
    if dir == 0:
        board.digital_pin_write(pin1, 1)
        board.digital_pin_write(pin2, 0)
    elif dir == 1:
        board.digital_pin_write(pin1, 0)
        board.digital_pin_write(pin2, 1)

def motorOnForSeconds(board, speed, sec, speedPin, pin1, pin2, dir):
    
    board.pwm_write(speedPin, speed)
    if dir == 0:
        board.digital_pin_write(pin1, 1)
        board.digital_pin_write(pin2, 0)
    elif dir == 1:
        board.digital_pin_write(pin1, 0)
        board.digital_pin_write(pin2, 1)
    time.sleep(sec)
    board.digital_pin_write(pin1, 0)
    board.digital_pin_write(pin2, 0)

def stepperMotorOnForDegrees(board, stepPin, dirPin,degrees,dir):
    board.digital_write(stepPin, dir)
    degree = 200/360
    print(degree)
    degree = (degree*5) * degrees
    print(degree)
    
    x=0
    while x <= degree:
    
     #DirPin High

        board.digital_pin_write(dirPin, 1)
        time.sleep(0.00035)
        board.digital_pin_write(dirPin, 0)
    
        x=x+1

board=NULL
board = contect2Ardunio(board)
board.set_pin_mode_digital_output(4) #Drive Pin
board.set_pin_mode_digital_output(3)
board.set_pin_mode_digital_output(6)
board.set_pin_mode_digital_output(7)
board.set_pin_mode_pwm_output(10)

stepperMotorOnForDegrees(board, 4, 3, 180, 1)
stepperMotorOnForDegrees(board, 4, 3, 180, 0)