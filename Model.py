import cv2
import uuid
import os
import time
import winsound

labels = ['penny', 'dollar']
number_imgs = 20

IMAGES_PATH = os.path.join('workspace', 'images', 'collectedimages')

if not os.path.exists(IMAGES_PATH):
    if os.name == 'posix':
        os.mkdir -p (IMAGES_PATH)
    if os.name == 'nt':
         os.mkdir (IMAGES_PATH)
for label in labels:
    path = os.path.join(IMAGES_PATH, label)
    if not os.path.exists(path):
        os.mkdir (path)

for label in labels:
    cap = cv2.VideoCapture(1)
    print('Collecting images for ()'.format(label))
    time.sleep(5)
    for imgnum in range(number_imgs):
        x =0
        print(imgnum)
        print('Collecting image ()'.format(imgnum))
        ret, frame = cap.read()
        imgname = os.path.join(IMAGES_PATH,label,label + str(imgnum)+'.jpg')
        cv2.imwrite(imgname, frame)
        cv2.imshow('frame', frame)
        winsound.Beep(440,500)
        time.sleep(2)
        x+1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()