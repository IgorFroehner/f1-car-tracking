import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from collections.abc import Iterable

def match_template(img, templates, method, output_path="output.png", show=False):
    img = img.copy()

    if type(templates) == list:
        ress = []
        vals = []
        whs = []
        for templ in templates:
            res = apply_method(img, templ, method)
            ress.append(res)
            vals.append(cv.minMaxLoc(res))
            whs.append(templ.shape[::-1])

        val = max(vals, key= lambda x: x[1])
        min_val, max_val, min_loc, max_loc = val

        res = ress[vals.index(val)]
        w, h = whs[vals.index(val)]
        
        # print(vals.index(val))
        # print(vals)
        # print(ress)
        # print(whs)
        
        # print(val)
        # print(res)
        # print(w, h)
        
        exit(0)
    else:
        template = templates

        w, h = template.shape[::-1]
        res = apply_method(img, template, method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(img,top_left, bottom_right, 255, 2)
    
    plt.subplot(121),plt.imshow(res, cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])

    plt.subplot(122),plt.imshow(img, cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    
    plt.suptitle(method + f'\nmin: {min_val}, max: {max_val}')
    plt.savefig(output_path)

    if show:
        plt.show()

def return_tracked_image(image, templates, method, colored_image):
    img = image.copy()

    ress = []
    vals = []
    whs = []
    for templ in templates:
        res = apply_method(img, templ, method)
        ress.append(res)
        vals.append(cv.minMaxLoc(res))
        whs.append(templ.shape[::-1])

    
    val = max(vals, key= lambda x: x[1])
    min_val, max_val, min_loc, max_loc = val
    
    res = ress[vals.index(val)]
    w, h = whs[vals.index(val)]

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    
    if max_val > 0.7:
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv.rectangle(colored_image,top_left, bottom_right, color=(0,0,255), thickness=2)

    return colored_image

def apply_method(img, template, method):
    new_image = img.copy()
    method = eval(method)

    # Apply template Matching
    res = cv.matchTemplate(new_image, template, method)
    
    return res

def get_templates():
    templates = []
    for i in range(1, 7):
        template_name = f'templates/rear{i}.png'
        templates.append(cv.imread(template_name, 0))
    return templates
