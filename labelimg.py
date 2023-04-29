import cv2 
import os
import time

SaveLocation = "C:/Users/break/Documents/Python/Penny/RotationTest"
Image2Save = cv2.imread("C:/Users/break/Documents/Python/CoinPictures3/CoinPictures3_000001.png")
saveFile = os.path.join(SaveLocation, "0")
print(saveFile)
print("C:/Users/break/Documents/Python/Penny/RotationTest/0")
cv2.imwrite(os.path.join(saveFile, str(time.time()) + ".png"), Image2Save)
print(os.path.join(saveFile, str(time.time()) + ".png"))


SaveLocation = "C:/Users/break/Documents/Python/Penny/RotationTest/0"
cv2.imwrite(os.path.join(saveFile, str(time.time()) + ".png"), Image2Save)

x=0
y=0
CATEGORIES = []