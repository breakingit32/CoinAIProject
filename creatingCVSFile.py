import numpy as np 
import pandas as pd 
import random

import cv2
import os


data_path = 'C://Users//break//Documents//Python//Penny//Rotation 28x28'
#data_path = os.path.join(root_dir,data_folder)
data_dir_list = os.listdir(data_path)
print ('the data list is: ',data_dir_list)

num_classes = 90
labels_name={'0':0, '4':1, '8':2, '12':3, '16':4, '20':5, '24':6, 
             '28':7, '32':8, '36':9, '40':10, '44':11, '48':12, '52':13, 
             '56':14, '60':15, '64':16, '68':17, '72':18, '76':19, '80':20,
             '84':21, '88':22, '92':23, '96':24, '100':25, '104':26, '108':27,
             '112':28, '116':29, '120':30, '124':31, '128':32, '132':33, '136':34,
             '140':35, '144':36, '148':37, '152':38, '156':39, '160':40, '164':41,
             '168':42, '172':43, '176':44, '180':45, '184':46, '188':47, '192':48,
             '196':49, '200':50, '204':51, '208':52, '212':53, '216':54, '220':55,
             '224':56, '228':57, '232':58, '236':59, '240':60, '244':61, '248':62,
             '252':63, '256':64, '260':65, '264':66, '268':67, '272':68, '276':69, 
             '280':70, '284':71, '288':72, '292':73, '296':74, '300':75, '304':76, 
             '308':77, '312':78, '316':79, '320':80, '324':81, '328':82, '332':83,
             '336':84, '340':85, '344':86, '348':87, '352':88, '356':89}

train_df = pd.DataFrame(columns=['FileName', 'Label', 'ClassName'])
test_df = pd.DataFrame(columns=['FileName', 'Label', 'ClassName'])

#number of images to take for test data from each flower category
num_images_for_test = 100

# Here data_dir_list = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
# Loop over every flower category
for dataset in data_dir_list:
    # load the list of image names in each of the flower category
    img_list = os.listdir(os.path.join(data_path,dataset))
    print ('Loading the images of dataset-'+'{}//n'.format(dataset))
    label = labels_name[dataset]
    num_img_files = len(img_list)
    num_corrupted_files=0
    test_list_index = random.sample(range(1, num_img_files-1), num_images_for_test)
    
    # read each file and if it is corrupted exclude it , if not include it in either train or test data frames
    for i in range(num_img_files):
        img_name = img_list[i]
        img_filename = os.path.join(data_path,dataset,img_name)
        
        try:
            input_img = cv2.imread(img_filename)
            img_shape=input_img.shape
            if i in test_list_index:
                test_df = test_df.append({'FileName': img_filename, 'Label': label,'ClassName': dataset},ignore_index=True)
            else:
                train_df = train_df.append({'FileName': img_filename, 'Label': label,'ClassName': dataset},ignore_index=True)       
        except:
            print ('{} is corrupted//n'.format(img_filename))
            num_corrupted_files+=1
    
    
    
    print ('Read {0} images out of {1} images from data dir {2}//n'.format(num_img_files-num_corrupted_files,num_img_files,dataset))

print ('completed reading all the image files and assigned labels accordingly')

if not os.path.exists('data_files'):
    os.mkdir('data_files')

train_df.to_csv('data_files/rotationTraining.csv')
test_df.to_csv('data_files/rotationTesting.csv')
print('The train and test csv files are saved')