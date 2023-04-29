
import cv2
import numpy as np

# Load the image
img = cv2.imread("E:\CoinCropping\Inputs/1.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply a median blur filter to the image
gray = cv2.medianBlur(gray, 5)
# Find the contours in the image
contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour
largest_contour = max(contours, key=cv2.contourArea)

# Get the bounding box of the largest contour
x, y, w, h = cv2.boundingRect(largest_contour)

# Crop the image to the bounding box of the largest contour
cropped = img[y:y+h, x:x+w]

# Create a mask with the same size as the cropped image and filled with black color(0)
mask = np.zeros((cropped.shape[0], cropped.shape[1]), dtype=np.uint8)

# Draw the largest contour on the mask with white color(255)
cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), -1)

# Use the mask to crop the image
cropped[np.where(mask==0)] = 0

# Create a black image with the same size as the cropped image
black_img = np.zeros_like(cropped)

# Combine the black image with the cropped image using the mask
final_img = cv2.bitwise_or(black_img, cropped, mask=mask)
cv2.imshow('d', final_img)
cv2.waitKey(0)
# Save the result
cv2.imwrite("cropped_image.jpg", final_img)