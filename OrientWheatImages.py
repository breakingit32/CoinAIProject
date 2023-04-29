import os
import numpy as np
import pathlib
import cv2 as cv
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# reduce system log presentation in console.
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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





#
# # Starting File location - All files - Organized in folders NAMED with Classifications.
# data_dir = "E:/PROJECTS/CentSearch/Sorted_Images/Reverse_Wheat/training/"
# data_dir = pathlib.Path(data_dir)
# print(data_dir)
# image_count = len(list(data_dir.glob('*/*.jpg')))
# print(image_count)
#
# img_height = 100
# img_width = 100
# batch_size = 2
#
# # ==================================================== #
# #             Using dataset_from_directory             #
# # ==================================================== #
# ds_train = tf.keras.preprocessing.image_dataset_from_directory(
#     data_dir,
#     labels="inferred",
#     label_mode="int",  # int, categorical, binary
#     # class_names=['obverse_linc_142', 'obverse_linc_16', . . . etc]
#     color_mode="grayscale",
#     batch_size=batch_size,
#     image_size=(img_height, img_width),  # reshape if not in this size
#     shuffle=True,
#     seed=123,
#     validation_split=0.1,
#     subset="training",
# )
#
# ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
#     data_dir,
#     labels="inferred",
#     label_mode="int",  # int, categorical, binary
#     # class_names=['obverse_linc_142', 'obverse_linc_16', . . . etc]
#     color_mode="grayscale",
#     batch_size=batch_size,
#     image_size=(img_height, img_width),  # reshape if not in this size
#     shuffle=True,
#     seed=123,
#     validation_split=0.1,
#     subset="validation",
# )



class_names = (['0', '1', '10', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '11', '110', '111',
               '112', '113', '114', '115', '116', '117', '118', '119', '12', '120', '121', '122', '123', '124', '125',
               '126', '127', '128', '129', '13', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139',
               '14', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '15', '150', '151', '152',
               '153', '154', '155', '156', '157', '158', '159', '16', '160', '161', '162', '163', '164', '165', '166',
               '167', '168', '169', '17', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '18',
               '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '19', '190', '191', '192', '193',
               '194', '195', '196', '197', '198', '199', '2', '20', '200', '201', '202', '203', '204', '205', '206',
               '207', '208', '209', '21', '210', '211', '212', '213', '214', '215', '216', '217', '218', '219', '22',
               '220', '221', '222', '223', '224', '225', '226', '227', '228', '229', '23', '230', '231', '232', '233',
               '234', '235', '236', '237', '238', '239', '24', '240', '241', '242', '243', '244', '245', '246', '247',
               '248', '249', '25', '250', '251', '252', '253', '254', '255', '256', '257', '258', '259', '26', '260',
               '261', '262', '263', '264', '265', '266', '267', '268', '269', '27', '270', '271', '272', '273', '274',
               '275', '276', '277', '278', '279', '28', '280', '281', '282', '283', '284', '285', '286', '287', '288',
               '289', '29', '290', '291', '292', '293', '294', '295', '296', '297', '298', '299', '3', '30', '300',
               '301', '302', '303', '304', '305', '306', '307', '308', '309', '31', '310', '311', '312', '313', '314',
               '315', '316', '317', '318', '319', '32', '320', '321', '322', '323', '324', '325', '326', '327', '328',
               '329', '33', '330', '331', '332', '333', '334', '335', '336', '337', '338', '339', '34', '340', '341',
               '342', '343', '344', '345', '346', '347', '348', '349', '35', '350', '351', '352', '353', '354', '355',
               '356', '357', '358', '359', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47',
               '48', '49', '5', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6', '60', '61', '62', '63',
               '64', '65', '66', '67', '68', '69', '7', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '8',
               '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '9', '90', '91', '92', '93', '94', '95',
               '96', '97', '98', '99'])


# model_name = 'cent_reverse_model_part/'
# Load model from previous epochs
model = keras.models.load_model('cent_reverse_model_part/')



# Array containing all the FOLDER names - Index is output of Network
# class_names = ds_train.class_names
print(class_names)

# Number fo Classifications is the number of elements in the above array.
num_classes = len(class_names)

# # Vary the brightness of the images to create more training data elements.
# def augment(x, y):
#     image = tf.image.random_brightness(x, max_delta=0.05)
#     return image, y
#
#
# ds_train = ds_train.map(augment)
#
# # Input 100 px square GRAYSCALE image - Output index location of CLASS_NAMES array.
# model = keras.Sequential(
#     [
#         layers.Input((100, 100, 1)),
#         layers.Conv2D(128, 3, padding="same"),
#         layers.Conv2D(64, 3, padding="same"),
#         layers.Conv2D(32, 3, padding="same"),
#         layers.Conv2D(16, 3, padding="same"),
#         layers.MaxPooling2D(),
#         layers.Flatten(),
#         layers.Dense(360),
#     ]
# )
#
#
# model.compile(
#     optimizer=keras.optimizers.Adam(),
#     loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True),],
#     metrics=["accuracy"],
# )
#
# # Load model from previous epochs
# model = keras.models.load_model('cent_reverse_model_part/')
#
# # Output model structure summary to console
# model.summary()
#
# # Output status bar during epochs
# AUTOTUNE = tf.data.AUTOTUNE
#
# train_ds = ds_train.cache().prefetch(buffer_size=AUTOTUNE)
# val_ds = ds_validation.cache().prefetch(buffer_size=AUTOTUNE)

# # Train and write model data after each epoch
# model.fit(
#     ds_train,
#     validation_data=ds_validation,
#     epochs=12,
#     callbacks=[keras.callbacks.ModelCheckpoint("cent_reverse_model_part")]
# )
#
# # Save model again after all epochs have run
# model.save('cent_reverse_model_part/')




# # Test image to check against the model.
# coin_image_path = 'images/wheat_reverse-18_58.jpg'
#
# # Open image as GRAYSCALE - Default is RGB
# img = tf.keras.utils.load_img(
#     coin_image_path, color_mode='grayscale', target_size=(img_height, img_width)
# )
#



# FILE WE'RE TESTING
# read original image as color
# start_url = 'E:/PROJECTS/CentSearch/Sorted_Images/1113/'

start_url = 'E:/PROJECTS/CentSearch/Sorted_Images/Reverse_Wheat/2B_oriented/'
file_count = 0  # Increment the file count as you iterate through the directory

mask_url = 'E:/PROJECTS/CentSearch/MasksAndIdeals/Wheat_Reverse_Mask_100px.jpg'
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
    # color_300px = resized[y2:h2, x2:w2]  # Resized is the Unblurred Image
    # color_300px = squarify(color_300px, 300)  # Color version of original image - Rescaled but NOT rotated

    # Create small GRAYSCALE IMAGE for training
    gray_100px = marked_image[y2:h2, x2:w2]
    gray_100px = squarify(gray_100px, 100)
    gray_100px = cv.cvtColor(gray_100px, cv.COLOR_BGR2GRAY)
    gray_100px = mask_img(gray_100px, mask)

    img_array = tf.keras.utils.img_to_array(gray_100px)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    # Calculate prediction from image provided
    predictions = model.predict(img_array)

    score = tf.nn.softmax(predictions[0])

    # Output the results
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

    rotation_angle = class_names[np.argmax(score)]
    rotation_angle = rotation_angle.replace("reverse_wheat_", "")
    print(f"Rotation Angle = {rotation_angle}\n")

    rotated = rotate(current_image, int(rotation_angle)*-1)
    # FileName = entry.replace(".jpg", "")
    cv2.imwrite(
        f"E:\\PROJECTS\\CentSearch\\Sorted_Images\\Reverse_Wheat\\2B_oriented\\{entry}",
        rotated)

# cv.imshow('Final Image', rotated)  # Shows just the mask used

cv.waitKey(0)
cv.destroyAllWindows()