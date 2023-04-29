import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Use CUDA-accelerated version of OpenCV
cv2.setUseOptimized(True)
cv2.setNumThreads(128)

DIR = "E:\CoinCropping\Inputs"

def process_image(filename):
    # Initialize attempt number and the blur and Hough circle parameters
    attempt = 1
    blur = 37
    param1 = 50
    param2 = 30

    while attempt <= 3:
        try:
            print(filename)
            # Loading image
            img = cv2.imread(os.path.join(DIR, filename))
            # Convert the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.medianBlur(gray, blur)
            # Use the HoughCircles function to detect circles in the image
            circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 1000, param1=param1, param2=param2, minRadius=1600, maxRadius=2000)

            # Get the largest circle
            largest_circle = max(circles[0], key=lambda x: x[2])

            # Get the coordinates and radius of the largest circle
            x, y, r = largest_circle

            # Crop the image to the bounding box of the largest circle
            cropped = img[int(y)-int(r):int(y)+int(r), int(x)-int(r):int(x)+int(r)]

            # Create a mask with the same size as the cropped image and filled with black color(0)
            mask = np.zeros((cropped.shape[0], cropped.shape[1]), dtype=np.uint8)

            # Draw the largest circle on the mask with white color(255)
            cv2.circle(mask, (int(r), int(r)), int(r), (255, 255, 255), -1)

            # Use the mask to crop the image
            cropped[np.where(mask==0)] = 0
            path = r"E:/CoinCropping"
            cv2.imwrite(os.path.join(os.path.join(path, "Saved"), filename + ".jpeg"), cropped)
            break
        except Exception as e:
            print(f"An error occurred while processing {filename}: {e}")
            if attempt == 1:
                blur += 10
            elif attempt == 2:
                param1 += 10
                param2 += 10
            else:
                # Save the failed file in the 'fails' folder
                path = r"E:/CoinCropping/fails"
                cv2.imwrite(os.path.join(path, filename + ".jpeg"), img)
                break
            attempt += 1
def main():
    sTime = time.time()
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, filename) for filename in os.listdir(DIR)]
        for future in as_completed(futures):
            future.result()
    eTime = time.time()
    print(eTime-sTime)
if __name__ == '__main__':
    main()



