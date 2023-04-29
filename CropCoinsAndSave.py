import os
import numpy as np
import cv2 as cv


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

# # THIS CODE IS NOT CURRENTLY USED IN THIS SCRIPT
# # Rotate image and Count White Pixels
# # Use IDEAL_CANNY as properly oriented TEST
# # Use source iIMAGE as mask, rotate the IMAGE until it aligns with the IDEAL_CANNY
# # The angle with the most white pixels wins
# def rotate_and_test(mask, img, good_count=0, good_angle=0):
#     for x in range(0, 360, 1):
#         rotated = rotate(img, x)
#         result = mask_img(rotated, mask)
#
#         count_result = cv.countNonZero(result)
#
#         if count_result > good_count:
#             good_count = count_result
#             good_angle = x
#             good_result = result
#
#     return good_count, good_angle, good_result


# Locate and Circle the coin(s) in the resized BGR image
def locate_coin(img):
    # Blur for circle find
    blur = cv.medianBlur(img, 7)
    blur = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)

    # Hough Circle Transform - Tutorial: https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html
    # Documentation: https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d
    circles = cv.HoughCircles(blur, cv.HOUGH_GRADIENT, 1, 400, param1=50, param2=30, minRadius=120, maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle in GREEN
        cv.circle(blur, (i[0], i[1]), i[2], (0, 255, 0), 2)

        # draw the center of the circle in RED
        cv.circle(blur,(i[0],i[1]),2,(0,0,255),2)
        # BLUR is the blurred and marked image

        # i[0] = x coordinate of the center of the circle
        # i[1] = y coordinate of the center of the circle
        # i[2] = radius of the circle

        # Calculate X, Y, W, H using the center of the circle and radius.
        # These dimensions will be used for cropping the grayscale image for recognition
        x = int(i[0]) - int(i[2])
        y = int(i[1]) - int(i[2])
        w = int(i[0]) + int(i[2])
        h = int(i[1]) + int(i[2])

        # Ensure X, Y, W, H won't overun the edge 0f the image
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if w > int(blur.shape[1]):
            w = int(blur.shape[1])
        if h > int(blur.shape[0]):
            h = int(blur.shape[0])

            # Calculate X, Y, W, H using the center of the circle and radius.
        # These dimensions will be used for cropping the color image for final rotation and markup.
        x2 = x - 10
        y2 = y - 10
        w2 = w + 10
        h2 = h + 10

        # Ensure X2, Y2, W2, H2 won't overun the edge of the image
        if x2 < 0:
            x2 = 0
        if y2 < 0:
            y2 = 0
        if w2 > int(blur.shape[1]):
            w2 = int(blur.shape[1])
        if h2 > int(blur.shape[0]):
            h2 = int(blur.shape[0])

    return img, x, y, w, h, x2, y2, w2, h2


def prep_image(img, mask):
    # img = cv2.imread(img_path)
    img = rescaleFrame(img, 400)
    resized = img
    img, x, y, w, h, x2, y2, w2, h2 = locate_coin(img)
    img = resized[y2:h2, x2:w2]  # Resized is the Unmarked Image
    img = squarify(img, 300)  # Color version of original image - Rescaled but NOT rotated
    image_array_100px = mask_img(img, mask)
    image_array_100px = squarify(image_array_100px, 100)
    image_array_100px = cv.cvtColor(image_array_100px, cv.COLOR_BGR2GRAY)
    
    return image_array_100px, img  # We return the unmasked image




    # # Output the results
    # print(
    #     "This image most likely belongs to {} with a {:.2f} percent confidence."
    #         .format(class_names[np.argmax(score)], 100 * np.max(score))
    # )

    return class_name, percentage


# def main():

# THE FOLLOWING TRAVERSAL WORKS WHEN THE ROOT FOLDER CONTAINS FOLDERS (NO FILES)
# AND THE SUB-FOLDERS CONTAIN FILES BUT NO SUB-FOLDERS THEMSELVES.
# I WAS PICTURING "YEAR" FOLDERS WITH "JPG" IMAGES INSIDE THEM

filecount = 0

# dir_name = 'E:/PROJECTS/CentSearch/test/'
dir_name = 'C:/Users/break/Documents/Python/TheCoinBot-main'


list_of_folders = os.listdir(dir_name)  # ARRAY OF SUB-FOLDER NAMES

# print(f'LIST OF FOLDERS - {list_of_folders}')

for folder in list_of_folders:


    # print(f'FOLDER - {folder}')  # FOLDER IS THIS SUB FOLDER
    # Create full path
    full_path = os.path.join(os.getcwd(),  folder)
    print(f'FULL PATH - {full_path}')
    list_of_files = os.listdir('C:/Users/break/Documents/Python/TheCoinBot-main/Picture')  # ARRAY OF FILE NAMES IN THIS SUB-FOLDER
    # print(f'LIST OF FILES - {list_of_files}')
    for file in list_of_files:

        print(f'FILENAME - {file}')
        item_id = file[0:8]  # eBay item-id is the first 8 characters of the filename
        print(f'ITEM ID - {item_id}')

        # Hough Circle Transform - Tutorial: https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html
        # DocS: https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d

        mask_url = 'C:/Users/break/Documents/Python/Wheat_Reverse_Mask_300px.jpg'
        mask = cv.imread(mask_url, 1)  # Filename in this directory
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

        img_height = 100
        img_width = 100
        batch_size = 2

        img_fullpath = full_path + "/" + file
        print(f'FILEPATH - {img_fullpath}')

        img = cv.imread('C:/Users/break/Documents/Python/TheCoinBot-main/Picture/123123123.JPG')
        
        
        x, y, w, h, x2, y2, w2, h2 = locate_coin(img)
        cv.circle(img, (w/2, h/2), 25, (255,0,0), -1)
        cv.imshow('img', img)
        cv.waitKey(1000)
        img = rescaleFrame(img, 800)

        # We add 50 pixels to the dimensions of the image to ensure we account
        # for images where the coin is cropped very close to the edge.
        h, w = img.shape[:2]
        AddSides = np.zeros((h + 50, w + 50, 3), np.uint8)
        AddSides[25:25 + h, 25:25 + w, :3] = img
        # cv.imshow(f'Added Sides {count}', AddSides)  # Shows just the mask used
        img = AddSides

        # cv.imshow('AddSides', img)

        max_radius = img.shape[0]
        print(max_radius)
        print(img.shape[1])

        cimg = img  # cv.cvtColor(img, cv.COLOR_GRAY2BGR)

        img = cv.medianBlur(img, 37)  # Higher blurring improves accuracy when there is higher contrast between coin and BKG
        # cv.imshow('Blur', img)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        max_radius = (min(img.shape[:2]))/2  # MaxRadius for the found circles is one half the smaller of the sides of the image

        circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 250,
                                  param1=50, param2=30, minRadius=50, maxRadius=int(max_radius))

        if circles is not None:

            print(f"!!!!!!!!!!!!!!!    Circles = {circles}")

            print(f"Number of Circles = {len(circles)}")

            circles = np.uint16(np.around(circles))

            imagecount = 0
            try:

                for i in circles[0, :]:
                    # # draw the outer circle
                    # cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    # # draw the center of the circle
                    # cv.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

                    # IMG is BLURRED and MARKED
                    # CIMG is the COPIED IMAGE that is NOT BLURRED and UNMARKED
                    # cv.imshow(f'Image', cimg)

                    x = int(i[0]) - int(i[2]) - 20
                    y = int(i[1]) - int(i[2]) - 20
                    w = int(i[0]) + int(i[2]) + 20
                    h = int(i[1]) + int(i[2]) + 20

                    # We no longer need to check for image overrun, since we added a 50 pixel border to every image

                    img_i = cimg[y:h, x:w]
                    img_i = squarify(img_i, 300)  # Color version of original image - Rescaled but NOT rotated

                    # cv.imshow(f'Image {i}', img_i)

                    print(f"Processing Filename = {file}")

                    

                    # image_path = start_url+entry
                    # current_img = cv2.imread(image_path)

                    img_array, current_img = prep_image(img_i, mask) # img_array is 100px grayscale - img is 300px color
                    

                    # # Output the results
                    # print(
                    #     f"Image {entry} most likely belongs to {coin_side} with a {certainty:.2f} percent confidence."
                    # )

                    angle_names = (['0', '1', '10', '100', '101', '102', '103', '104', '105', '106', '107', '108',
                                    '109', '11',
                                    '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '12',
                                    '120', '121',
                                    '122', '123', '124', '125', '126', '127', '128', '129', '13', '130', '131',
                                    '132', '133',
                                    '134', '135', '136', '137', '138', '139', '14', '140', '141', '142', '143',
                                    '144', '145',
                                    '146', '147', '148', '149', '15', '150', '151', '152', '153', '154', '155',
                                    '156', '157',
                                    '158', '159', '16', '160', '161', '162', '163', '164', '165', '166', '167',
                                    '168', '169',
                                    '17', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179',
                                    '18', '180',
                                    '181', '182', '183', '184', '185', '186', '187', '188', '189', '19', '190',
                                    '191', '192',
                                    '193', '194', '195', '196', '197', '198', '199', '2', '20', '200', '201',
                                    '202', '203',
                                    '204', '205', '206', '207', '208', '209', '21', '210', '211', '212', '213',
                                    '214', '215',
                                    '216', '217', '218', '219', '22', '220', '221', '222', '223', '224', '225',
                                    '226', '227',
                                    '228', '229', '23', '230', '231', '232', '233', '234', '235', '236', '237',
                                    '238', '239',
                                    '24', '240', '241', '242', '243', '244', '245', '246', '247', '248', '249',
                                    '25', '250',
                                    '251', '252', '253', '254', '255', '256', '257', '258', '259', '26', '260',
                                    '261', '262',
                                    '263', '264', '265', '266', '267', '268', '269', '27', '270', '271', '272',
                                    '273', '274',
                                    '275', '276', '277', '278', '279', '28', '280', '281', '282', '283', '284',
                                    '285', '286',
                                    '287', '288', '289', '29', '290', '291', '292', '293', '294', '295', '296',
                                    '297', '298',
                                    '299', '3', '30', '300', '301', '302', '303', '304', '305', '306', '307',
                                    '308', '309',
                                    '31', '310', '311', '312', '313', '314', '315', '316', '317', '318', '319',
                                    '32', '320',
                                    '321', '322', '323', '324', '325', '326', '327', '328', '329', '33', '330',
                                    '331', '332',
                                    '333', '334', '335', '336', '337', '338', '339', '34', '340', '341', '342',
                                    '343', '344',
                                    '345', '346', '347', '348', '349', '35', '350', '351', '352', '353', '354',
                                    '355', '356',
                                    '357', '358', '359', '36', '37', '38', '39', '4', '40', '41', '42', '43',
                                    '44', '45', '46',
                                    '47', '48', '49', '5', '50', '51', '52', '53', '54', '55', '56', '57', '58',
                                    '59', '6', '60',
                                    '61', '62', '63', '64', '65', '66', '67', '68', '69', '7', '70', '71', '72',
                                    '73', '74', '75',
                                    '76', '77', '78', '79', '8', '80', '81', '82', '83', '84', '85', '86', '87',
                                    '88', '89', '9',
                                    '90', '91', '92', '93', '94', '95', '96', '97', '98', '99'])

                    
                    cv.imshow('yest', current_img)
                    cv.waitKey(1000)
                    filecount = filecount + 1

            except Exception as e:
                print(f"\nHTTP Error 504: Gateway Time-out = {e}")
                print("urllib.error.HTTPError: HTTP Error 504: Gateway Time-out - ATTEMPTING TO CONTINUE.")


cv.waitKey(0)
cv.destroyAllWindows()


# if __name__ == '__main__':
#     main()