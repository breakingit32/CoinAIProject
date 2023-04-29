import os
from PIL import Image

image = Image.open('C:/Users/break/Documents/Python/Penny/TestPenny.png')
width, height = image.size
print(width)
print(height)
nDark = 0
nLite = 0
thresh = 0.33
for x in range(width):
    for y in range(height):
        r,g,b = image.getpixel((x,y))
        #print(r,g,b)
        sum =r + g + b
        if (sum > thresh*255*3):
            nLite = nLite + 1
            image.putpixel((x,y), (255,255,255))
        else:
            nDark = nDark + 1
            image.putpixel((x,y), (0))
image.show("yes")
print (nLite, " ", nDark)
print (float(nDark) / float((nLite + nDark)))