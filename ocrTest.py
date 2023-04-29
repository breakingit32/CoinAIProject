
import numpy as np
import cv2
import os
import easyocr

print(cv2.__version__)
reader = easyocr.Reader(['en'], gpu=False)
 
myMask="C:/Users/break/Documents/Python/DateMask.jpg"
imgDir = "C:/Users/break/Documents/Python/Penny/Up/Penny761661103.png"
testDir ="C:/Users/break/Documents/Python/Penny/1000x1000"
#myMask=cv2.add(myMask,myMask2)
#myMask=np.logical_or(myMask,myMask2)
myMask = cv2.imread(myMask, 1)

imgSize=400
adjust = 1.33333333333333333333333
#myMask=cv2.bitwise_not(myMask)
img = cv2.imread(imgDir)
img = cv2.resize(img, (imgSize,imgSize))
mask = np.zeros(img.shape[:2], dtype="uint8")
cv2.rectangle(mask, (int(225*adjust), int(188*adjust)), (int(291*adjust),int(228*adjust)), (255,255,255), -1)
masked = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow('masked', masked)
cv2.imwrite(os.path.join(testDir, "Test.png"), masked)  

results = reader.readtext(masked)

print(results)





cv2.waitKey(0)
cv2.destroyAllWindows()


