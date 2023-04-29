import cv2
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
def renaming(Dir, base_name):
    

    path = os.getcwd()

    # path to the data folder
    data_path = 'C:/Users/break/Documents/Python/Penny/DigitExtractor'
    
    data_list = os.listdir(os.path.join(data_path,Dir))

    print (os.path.join(data_path,Dir))
    os.chdir(os.path.join(data_path,Dir))
    # The base name of image files
    
    for i in range(len(data_list)):
        img_name = data_list[i]
    #    img_rename = base_name + '_{}'.format(i+1)+'.png' # here the file name is base_name_1.png
        img_rename = base_name + '_{:06d}'.format(i+1)+'.png' # here the file name is base_name_000001.png
        if not os.path.exists(img_rename):
            os.rename(img_name,img_rename)
        
    os.chdir(path)

def cropped_contours(imgPath):

    img= cv2.imread(imgPath)
    cv2.waitKey(0)
    
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, 
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea,reverse= True)
    x, y, w, h = cv2.boundingRect(contours[0])
    cropped = img[y:y+h, x:x+w]
    cropped = cv2.resize(cropped, (20,32))
    return cropped


MainDir = 'C:/Users/break/Documents/Python/Penny/DigitDataSet'

for subDir in os.listdir(MainDir):
    path = os.path.join(MainDir, subDir)
    for img in os.listdir(path):
        imgPath = os.path.join(path, img)
        print(imgPath)
        result = cropped_contours(imgPath)
        cv2.imwrite(imgPath, result)
