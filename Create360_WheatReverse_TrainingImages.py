# IMPORTS
import numpy as np
import cv2 as cv
import cv2
import os


# FUNCTIONS
def mask_img(img, mask):
    result = img.copy()
    result[mask == 0] = 0
    result[mask != 0] = img[mask != 0]
    return result


# Rotation
def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2]

    if rotPoint is None:
        rotPoint = (width // 2, height // 2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width, height)

    return cv.warpAffine(img, rotMat, dimensions)


# Function to Make the shortest side of an image = MIN_DIMENSION
# Pass the source IMAGE -and- MIN_DIMENSION for both length and width
# This is to make larger images more manageable.
def rescaleFrame(frame, min_dimension):
    # Grab the width and height of the source image.
    width = int(frame.shape[1])
    height = int(frame.shape[0])

    if width == height:  # Original image is SQUARE
        newheight = min_dimension
        newwidth = min_dimension

    elif width > height:  # Original image is LANDSCAPE
        newwidth = width * (min_dimension / height)
        newheight = min_dimension

    else:  # Original image is PORTRAIT
        newheight = height * (min_dimension / width)
        newwidth = min_dimension

    dimensions = (int(newwidth), int(newheight))

    return cv.resize(frame, dimensions, interpolation=cv.INTER_CUBIC)  # or  INTER_AREA


# resize cropped image and make it a square
def squarify(frame, dimension):
    # Set height and width to dimension to create a square image.
    dimensions = (int(dimension), int(dimension))

    return cv.resize(frame, dimensions, interpolation=cv.INTER_CUBIC)  # or  INTER_AREA


# Rotate image and test sharpness
# Use IDEAL_CANNY as properly oriented TEST
# Use source iIMAGE as mask, rotate the IMAGE until it aligns with the IDEAL_CANNY
# The angle with the most white pixels wins
def rotate_and_test(mask, img, good_count=0, good_angle=0):
    for x in range(0, 360, 1):
        # print(x)
        rotated = rotate(img, x)
        # rotated = cv2.Canny(rotated, 140, 150)
        #        cv.imshow(f"Rotated Canny =  {x}", rotated)
        result = mask_img(rotated, mask)
        # result = cv.medianBlur(result,3)

        count_result = cv.countNonZero(result)
        # count_result = cv.Laplacian(result, cv.CV_64F).var()

        if count_result > good_count:
            good_count = count_result
            #           cv.imshow(f"Good Result =  {x}", result)

            #            cv.waitKey(0)
            #            cv.destroyAllWindows()

            #            print(f"Good Result =  {good_count}. Angle = {x}.")
            good_angle = x
            good_result = result

    return good_count, good_angle, good_result


# Locate and Circle the coin(s) in the resized BGR image
def locate_coin(img):
    # global count
    # count = count+1

    # We add 50 pixels to the dimensions of the image to ensure we account
    # for images where the coin is cropped very close to the edge.
    h, w = img.shape[:2]
    AddSides = np.zeros((h + 50, w + 50, 3), np.uint8)
    AddSides[25:25+h, 25:25+w, :3] = img
    # cv.imshow(f'Added Sides {count}', AddSides)  # Shows just the mask used
    img = AddSides

    # Blur for circle find
    blur = cv.medianBlur(img, 11)
    blur = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
    # cv.imshow(f'BLUR {count}', blur)  # Shows just the mask used

    # Hough Circle Transform - Tutorial: https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html
    # Documentation: https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d
    circles = cv.HoughCircles(blur, cv.HOUGH_GRADIENT, 1, 400, param1=50, param2=30, minRadius=120, maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        unmarked = img.copy()
        # draw the outer circle in GREEN
        cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)

        # draw the center of the circle in RED
        cv.circle(img,(i[0],i[1]),2,(0,0,255),2)

        # IMG is now he marked image

        # i[0] = x coordinate of the center of the circle
        # i[1] = y coordinate of the center of the circle
        # i[2] = radius of the circle

        # Calculate X, Y, W, H using the center of the circle and radius.
        # These dimensions will be used for cropping the grayscale image for recognition
        x = int(i[0]) - int(i[2])
        y = int(i[1]) - int(i[2])
        w = int(i[0]) + int(i[2])
        h = int(i[1]) + int(i[2])

        # We no longer need to check for image overrun, since we added a 50 pixel border to every image

        # Calculate X, Y, W, H using the center of the circle and radius.
        # These dimensions will be used for cropping the color image for final rotation and markup.
        x2 = x - 10
        y2 = y - 10
        w2 = w + 10
        h2 = h + 10

        # We no longer need to check for image overrun, since we added a 50 pixel border to every image

        # cv.imshow(f'IMG {count}', img)  # Shows the marked image

    return unmarked, x, y, w, h, x2, y2, w2, h2


# LOAD IMAGES AND MASKS

# FILE WE'RE TESTING
# read original image as color
count = 1

start_url = 'E:/PROJECTS/CentSearch/Sorted_Images/Reverse_Wheat/oriented/'
file_count = 0  # Increment the file count as you iterate through the directory

mask_url = 'E:/PROJECTS/CentSearch/MasksAndIdeals/Wheat_Reverse_Mask_300px.jpg'
mask = cv.imread(mask_url, 1)  # Filename in this directory
mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

listOfFile = os.listdir(start_url)
allFiles = list()
# Iterate over all the entries
for entry in listOfFile:
    # For each filename in this directory
    file_count += 1  # Increment file_count by 1
    print(f"Starting Image Number {file_count}")
    print(f"Filename = {entry}")

    current_image = cv.imread(start_url + entry, 1)  # Filename in this directory

    # start = cv.imread(start_url + entry, 1)  # Filename in this directory

    # RESIZE input image to manageable size - shortest side = 400px
    resized = rescaleFrame(current_image, 400)

    marked_image = resized.copy()
    marked_image, x, y, w, h, x2, y2, w2, h2 = locate_coin(marked_image)

    # # CROP and RESIZE
    # # Create 300px Color version for FINAL image
    color_300px = marked_image[y2:h2, x2:w2]  # Resized is the Unblurred Image
    color_300px = squarify(color_300px, 300)  # Color version of original image - Rescaled but NOT rotated
    color_300px = mask_img(color_300px, mask)

    # Create small GRAYSCALE IMAGE for training
    # gray_100px = resized[y2:h2, x2:w2]
    gray_100px = squarify(color_300px, 100)
    gray_100px = cv.cvtColor(gray_100px, cv.COLOR_BGR2GRAY)

    for x in range(0, 360, 1):
        # print(x)
        rotated = rotate(gray_100px, x)
        # rotated = mask_img(rotated, mask)
        FileName = entry.replace(".jpg", "")
        cv2.imwrite(
            f"E:/PROJECTS/CentSearch/training/reverse_wheat/angle/reverse_wheat_{x}/{file_count}-reverse_wheat_{x}.jpg",
            rotated)

cv.imshow('Final Image', rotated)  # Shows just the mask used

cv.waitKey(0)
cv.destroyAllWindows()
