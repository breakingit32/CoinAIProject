import cv2
import os
import time


DATADIR = "C:/Users/break/Documents/Python/Rotation Test/output"
Mask = ['C:/Users/break/Documents/Python/Rotation Test/firstDigitMask.png',
        'C:/Users/break/Documents/Python/Rotation Test/secondDigitMask.png',
        'C:/Users/break/Documents/Python/Rotation Test/thirdDightMask.png',
        'C:/Users/break/Documents/Python/Rotation Test/fourthDigitMask.png']
SaveLocations = ['C:/Users/break/Documents/Python/Penny/DigitExtractor/1First',
                 'C:/Users/break/Documents/Python/Penny/DigitExtractor/2Second',
                 'C:/Users/break/Documents/Python/Penny/DigitExtractor/3Third',
                 'C:/Users/break/Documents/Python/Penny/DigitExtractor/4Fourth']
y = 0
for img in os.listdir(DATADIR):
    x = 0
    
    for mask in Mask:
        image = cv2.imread(os.path.join(DATADIR, img))
        
        masks = cv2.imread(mask)
        masks = cv2.cvtColor(masks, cv2.COLOR_BGR2GRAY)
        masks = cv2.resize(masks, (300,300))
        print(masks.shape)
        print(image.shape)
        replaced_image = cv2.bitwise_and(image,image,mask = masks)
        cv2.imwrite(os.path.join(SaveLocations[x], str(time.time()) + ".png"), replaced_image)
        y = y + 1
        x = x + 1
