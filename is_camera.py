import numpy as np
import cv2

TEMPLATE_IMAGE_PATH = './047410018_box3_frame_045789.png'


def template_matching(gray):
    template = cv2.imread(TEMPLATE_IMAGE_PATH, 0)
    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return min_val, max_val, min_loc, max_loc


def is_camera(gray: np.ndarray, th_rate=0.6):
    min_val, max_val, min_loc, max_loc = template_matching(gray)
    if max_val >= th_rate:
        return True
    else:
        return False
