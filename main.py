import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from collections.abc import Iterable

from tests import create_tests
from video import create_video

if __name__ == '__main__':
    # All the 6 methods for comparison in a list
    methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
    
    # Create Tests
    # create_tests(methods=methods)

    # Create Video
    create_video('eau_rouge_easy', 'cv.TM_CCOEFF_NORMED')