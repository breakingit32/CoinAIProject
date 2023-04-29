import os
import cv2
import math
import time
import tkinter as tk
from tkinter import filedialog
import logging
from PIL import Image, ImageTk
import tkinter.ttk as ttk
from queue import Queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
logging.basicConfig(filename="coin_cropping.log", level=logging.INFO)

cv2.setUseOptimized(True)
cv2.setNumThreads(128)

DIR = "E:\CoinCropping\Inputs"

total_processing_time, num_processed_images = 0, 0


def process_image(filename, queue):
    attempt = 1
    blur = 37
    param1 = 50
    param2 = 30

    while attempt <= 3:
        try:
            print(filename)
            img = cv2.imread(os.path.join(DIR, filename))
            
            if img is None or img.size == 0:
                logging.error(f"Unable to read {filename}")
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.medianBlur(gray, blur)
            circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 1000, param1=param1, param2=param2, minRadius=1600, maxRadius=2000)

            largest_circle = max(circles[0], key=lambda x: x[2])

            x, y, r = largest_circle

            cropped = img[int(y)-int(r):int(y)+int(r), int(x)-int(r):int(x)+int(r)]

            mask = np.zeros((cropped.shape[0], cropped.shape[1]), dtype=np.uint8)

            cv2.circle(mask, (int(r), int(r)), int(r), (255, 255, 255), -1)

            cropped[np.where(mask == 0)] = 0
            path = r"E:/CoinCropping/Saved"
            cv2.imwrite(os.path.join(path, filename + ".jpeg"), cropped)
            queue.put((cropped, filename))
            break

        except Exception as e:
            logging.error(f"An error occurred while processing {filename}: {e}")
            if attempt == 1:
                blur += 10
            elif attempt == 2:
                param1 += 10
                param2 += 10
            else:
                path = r"E:/CoinCropping/fails"
                if img is not None and img.size != 0:
                    cv2.imwrite(os.path.join(path, filename + ".jpeg"), img)
                break
            attempt += 1
def submit_choices():
    approved = 0
    declined = 0

    for filename, approval in image_choices:
        if approval.get() == 1:
            approved += 1
            print(f"Image {filename} approved.")
        elif approval.get() == 2:
            declined += 1
            path = r"E:/CoinCropping/fails"
            cv2.imwrite(os.path.join(path, filename + ".jpeg"), img)
            print(f"Image {filename} declined.")
        else:
            print(f"No decision made for {filename}")

    num_approved.set(approved)
    progress_label.config(text=f"Approved: {approved}, Declined: {declined}")

def show_image(img, filename, img_label, approve_button, decline_button, r, c):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = img.resize((200, 200), Image.ANTIALIAS)  # Resize the image to 200x200 pixels
    img_tk = ImageTk.PhotoImage(img)
    img_label.config(image=img_tk)
    img_label.image = img_tk
    img_label.grid(row=r*2, column=c*2, padx=10, pady=10)

    approve_button.grid(row=r*2+1, column=c*2, padx=5, pady=5, sticky='e')
    decline_button.grid(row=r*2+1, column=(c*2)+1, padx=5, pady=5, sticky='w')

def process_images_thread(file_list):
    processed_images = []

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image_with_timeout, filename, Queue(), 10) for filename in file_list]
        for future in as_completed(futures):
            img_and_filename = future.result()
            if img_and_filename is not None:
                processed_images.append(img_and_filename)
                progress_bar["value"] += 1
                update_eta_label(len(file_list) - len(processed_images))

    show_all_images(processed_images)



def approve_image(img, filename):
    print(f"Image {filename} approved.")

def decline_image(img, filename):
    path = r"E:/CoinCropping/fails"
    cv2.imwrite(os.path.join(path, filename + ".jpeg"), img)
    print(f"Image {filename} declined.")

def create_config_folder():
    if not os.path.exists("config"):
        os.makedirs("config")

def read_last_run_time():
    create_config_folder()
    if not os.path.exists("config/last_run_time.txt"):
        with open("config/last_run_time.txt", "w") as f:
            f.write("0")
    with open("config/last_run_time.txt", "r") as f:
        timestamp = float(f.read())
    return timestamp

def update_last_run_time():
    create_config_folder()
    with open("config/last_run_time.txt", "w") as f:
        f.write(str(time.time()))

def read_scanned_files():
    create_config_folder()
    if not os.path.exists("config/scanned_files.txt"):
        with open("config/scanned_files.txt", "w") as f:
            pass
    with open("config/scanned_files.txt", "r") as f:
        return set(f.read().splitlines())

def update_scanned_files(file_list):
    create_config_folder()
    with open("config/scanned_files.txt", "a") as f:
        for filename in file_list:
            f.write(filename + "\n")
def process_and_show_image(filename, callback):
    queue = Queue()
    process_image(filename, queue)
    img, filename = queue.get()
    callback(img, filename)
def update_eta_label(remaining_files):
    global total_processing_time, num_processed_images

    if num_processed_images == 0:
        eta_text = "ETA: Calculating..."
    else:
        avg_time_per_image = total_processing_time / num_processed_images
        eta_seconds = avg_time_per_image * remaining_files
        eta_minutes, eta_seconds = divmod(eta_seconds, 60)
        eta_text = f"ETA: {int(eta_minutes)}m {int(eta_seconds)}s"

    eta_label.config(text=eta_text)
def show_all_images(processed_images):
    num_images = len(processed_images)
    columns = 3
    rows = math.ceil(num_images / columns)

    for i, (img, filename) in enumerate(processed_images):
        r, c = divmod(i, columns)

        img_label = tk.Label(frame)
        approval = tk.IntVar()

        approve_button = tk.Radiobutton(frame, text="Approve", variable=approval, value=1)
        decline_button = tk.Radiobutton(frame, text="Decline", variable=approval, value=2)

        show_image(img, filename, img_label, approve_button, decline_button, r, c)

        image_choices.append((filename, approval))

    submit_button = tk.Button(frame, text="Submit", command=submit_choices)
    submit_button.grid(row=rows*2, columnspan=columns, pady=5)



def main(file_list):
    if not file_list:
        progress_label.config(text="Processing completed!")
        return

    processing_thread = threading.Thread(target=process_images_thread, args=(file_list,))
    processing_thread.start()

def process_image_with_timeout(filename, queue, timeout):
    processing_thread = threading.Thread(target=process_image, args=(filename, queue))

    processing_thread.start()
    processing_thread.join(timeout)

    if processing_thread.is_alive():
        logging.warning(f"Processing for {filename} took too long. Bypassing this image.")
        return None
    else:
        return queue.get()

def run_main():
    scanned_files = read_scanned_files()
    all_files = set(os.listdir(DIR))

    new_files = all_files.difference(scanned_files)

    if not new_files:
        progress_label.config(text="No new images to process.")
        return

    file_count = len(new_files)
    progress_bar["maximum"] = file_count
    progress_bar["value"] = 0

    update_scanned_files(new_files)
    main(list(new_files))
def go_back():
    img_label.grid_remove()
    approve_button.grid_remove()
    decline_button.grid_remove()
    submit_button.grid_remove()

    progress_label.grid(row=2, column=0, pady=5, sticky='w')
    progress_bar.grid(row=3, column=0, sticky='ew')
    process_button.grid(row=1, column=0, pady=5, sticky='ew')

def reset_app():
    global DIR, image_choices, num_approved, approval

    DIR = ""  # Reset the DIR variable
    image_choices = []  # Reset the list of image choices
    num_approved.set(0)  # Reset the number of approved images
    approval.set(0)  # Reset the approval variable

    # Update the labels and progress bar
    dir_label.config(text=f"Selected Directory: {DIR}")
    progress_label.config(text="")
    eta_label.config(text="")
    progress_bar['value'] = 0

    # Hide the image, approve, and decline buttons
    img_label.grid_remove()
    approve_button.grid_remove()
    decline_button.grid_remove()

    # Show the progress bar and start processing button again
    progress_label.grid(row=2, column=0, pady=5, sticky='w')
    progress_bar.grid(row=3, column=0, sticky='ew')
    process_button.grid(row=1, column=0, pady=5, sticky='ew')

def browse_directory():
    global DIR
    DIR = filedialog.askdirectory()
    dir_label.config(text=f"Selected Directory: {DIR}")

root = tk.Tk()
root.title("Coin Cropping Application")
root.geometry("600x300")
frame = tk.Frame(root)
frame.grid(padx=10, pady=10)

dir_label = tk.Label(frame, text=f"Selected Directory: {DIR}")
dir_label.grid(row=0, column=0, columnspan=2, pady=5)

browse_button = tk.Button(frame, text="Browse Directory", command=browse_directory)
browse_button.grid(row=1, column=0, pady=5)

process_button = tk.Button(frame, text="Start Processing", command=run_main)
process_button.grid(row=1, column=1, pady=5)

progress_label = tk.Label(frame, text="")
progress_label.grid(row=2, column=0, columnspan=2, pady=5)

progress_bar = ttk.Progressbar(frame, mode='determinate')
progress_bar.grid(row=3, column=0, columnspan=2, sticky='ew')

eta_label = tk.Label(frame, text="")
eta_label.grid(row=4, column=0, columnspan=2, pady=5)

image_choices = []
num_approved = tk.IntVar()
img_label = tk.Label(frame)

approval = tk.IntVar()
approve_button = tk.Button(frame, text="Approve", command=lambda: approval.set(1))
decline_button = tk.Button(frame, text="Decline", command=lambda: approval.set(2))

submit_button = tk.Button(frame, text="Submit", command=go_back)
submit_button.grid(row=6, column=0, columnspan=2, pady=5)

root.mainloop()
