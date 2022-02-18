import cv2
import numpy as np
from pathlib import Path
import os

from functions import return_tracked_image, get_templates, match_template

def create_video(input_folder, method, video_name='project_easy'):
    templates = get_templates()
    image_list = os.listdir(f'./{input_folder}/')
    image_list = sorted(image_list)

    img_array = []

    for image in image_list:
        img = cv2.imread(f'./{input_folder}/{image}')
        height, width, layers = img.shape
        size = (width,height)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        tracked_img = return_tracked_image(img_gray, templates=templates, method=method, colored_image=img)
        img_array.append(tracked_img)

    video = cv2.VideoWriter(f'{video_name}.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    for tracked_image in img_array:
        video.write(tracked_image)
    
    video.release()