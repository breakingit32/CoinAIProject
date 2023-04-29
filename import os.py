import os


path = "C:/Users/break/Documents/Python/Penny/RotationTest"
x = 0
while x < 360:
    os.mkdir(os.path.join(path, str(x)))
    x = x + 4