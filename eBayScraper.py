# import mechanicalsoup
from bs4 import BeautifulSoup
import requests
import numpy as np
import cv2 as cv
import cv2
import os
import urllib.request
import unicodedata
import re
import time
import random


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    And - https://stackoverflow.com/questions/295135/turn-a-string-into-a-valid-filename/46801075
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


year = 1909 - 1

while year < 1919:
    year = year + 1
    page = 4

    while page < 10:
        page = page + 1

        print(f"Page = {str(page)}")

        start_url = (f"https://www.ebay.com/sch/i.html?_from=R40&_nkw={year}"
                     f"+lincoln+cent&_sacat=39455&_ipg=25&_pgn={page}&rt=nc")

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome / 80.0.3987.132Safari / 537.36'}
        # content = requests.get(self.search_url, headers=headers).text
        #
        # print(f"Start URL = {start_url}")

        # search_response = requests.get(start_url, headers=headers)
        search_response = requests.get(start_url)
        search_content = search_response.content
        search_parser = BeautifulSoup(search_content, 'html.parser')
        search_body = search_parser.body  # search_body is a STRING containing all the HTML from the URL page

        # print(f"BODY = {search_body}")

        items_imgdiv = search_body.find_all("div", {"class": "s-item__image"})  # items_imgdiv is an array of DIVS

        # print(items_imgdiv)       # items_imgdiv is an array of DIVS
        # print(len(items_imgdiv))  # Number of elements in the array

        for items in items_imgdiv:

            # The returned links contain a tracking ID and HASH after the URL - This needs to be cropped off
            # i.e. https://www.ebay.com/itm/362662822247?_trkparms=ispr%3D1&hash=item5470638167:g:1m4AAOSwqc5cmdat. . .

            item_url = items.find('a')['href']  # [0:33]   # Item URL
            item_id = items.find('a')['href'][25:33]  # Item ID
            item_desc = items.find('img')['alt']   # Item Description
            item_desc = slugify(item_desc)  # Slugify makes the description safe for the filename

            print("\n")
            print(item_url)
            print(f"Year = {str(year)} Page = {str(page)}")
            # print(item_desc)

            # # THE FOLLOWING CODE WORKS TO GRAB THE IMAGE THUMBNAILS FROM THE JAVA SLIDER

            response = requests.get(item_url)
            content = response.content

            parser = BeautifulSoup(content, 'html.parser')
            body = parser.body

            # print(body)
            # print(imgs_div)  # Prints DIV with all Thumbnail Images

            imgs_div = body.find_all("div", {"id": "vi_main_img_fs_slider"})
            # # THE CODE BELOW WORKS TO DOWNLOAD THE IMAGES IN EACH EBAY LISTING
            x = 0

            try:

                for image in imgs_div[0].find_all('img'):
                        tn_img = image.get('src')
                        # tn_img = image.find('img')['src']
                        lg_img = tn_img.replace("s-l64.jpg", "s-l1600.jpg")
                        print(f"     Large Image = {lg_img}")
                        file_path = (f"E:/PROJECTS/CentSearch/Internet_Images/"
                                     f"{str(year)}/{str(item_id)}_{str(item_desc)}_{str(x)}.jpg")
                        print(f"     Writing File = {str(year)} - {str(item_id)}_{str(item_desc)}_{str(x)}.jpg")
                        urllib.request.urlretrieve(lg_img, file_path)
                        random_seconds = random.randint(300, 2000)/1000
                        print(f"     Random Seconds = {random_seconds}")
                        time.sleep(random_seconds)
                        print(f"     Image = {str(x)}")
                        x = x + 1

            except IndexError as err:
                # ONLY ONE IMAGE IN THIS LISTING - ATTEMPTING TO CONTINUE.
                imgs_div = body.find_all("img", {"id": "icImg"})
                # print(f"\nIMAGES DIV = {imgs_div}")
                # print(f"\nIMAGES DIV [0] = {imgs_div[0]}")
                # x = 0
                print("IndexError: {0}".format(err))
                print("ONLY ONE IMAGE IN THIS LISTING - ATTEMPTING TO CONTINUE.")
                tn_img = imgs_div[0].get('src')[0:48]  # crop image name from end of URL
                lg_img = tn_img + "s-l1600.jpg"
                print(f"     Large Image = {lg_img}")
                file_path = (f"E:/PROJECTS/CentSearch/Internet_Images/"
                             f"{str(year)}/{str(item_id)}_{str(item_desc)}_{str(x)}.jpg")
                print(f"     Writing File = {str(year)} - {str(item_id)}_{str(item_desc)}_{str(x)}.jpg")
                urllib.request.urlretrieve(lg_img, file_path)
                random_seconds = random.randint(1000, 3000) / 1000
                print(f"     Random Seconds = {random_seconds}")
                time.sleep(random_seconds)
                print(f"     Image = {str(x)}")
                # break

            except Exception as e:
                print(f"\nHTTP Error 504: Gateway Time-out = {e}")
                print("urllib.error.HTTPError: HTTP Error 504: Gateway Time-out - ATTEMPTING TO CONTINUE.")
