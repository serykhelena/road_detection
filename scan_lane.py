
from __future__ import print_function
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import math
import pickle
import unicorn
from scipy import ndimage
import scipy
from skimage.filters import roberts, sobel

'''
with open('cal_param_opts.txt', 'rb') as fp:
    opts = pickle.load(fp)

with open('cal_param_ipts.txt', 'rb') as fp:
    ipts = pickle.load(fp)

image = cv2.imread('pics_one_lane_x/frame1_1.png')
undist_img = unicorn.undistort_image(image, opts, ipts)

warped = unicorn.bird_eye_view(undist_img)   # (maxWidth, maxHeight))
cv2.imshow('warepd', warped)
'''

warped = cv2.imread('output_images/warped1_1.png')
color_mask = unicorn.hsv_green_black_mask(warped)

# testing adaptive thresholding
gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
median_blur = cv2.medianBlur(gray_warped, 5)
# ret, th1 = cv2.threshold(median_blur, 127, 255, cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(median_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 3)
# th3 = cv2.adaptiveThreshold(median_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)


blur_warped = cv2.GaussianBlur(warped, (5, 5), 0)
# blur_warped = cv2.bilateralFilter(warped, 9, 75, 75)
hls_warped = cv2.cvtColor(blur_warped, cv2.COLOR_BGR2HLS)
h_ch, l_ch, s_ch = cv2.split(hls_warped)

# L channel processing
l_mag = unicorn.mag_thresh(l_ch, 5, (10, 250))
l_dir_mag = unicorn.dir_threshold(l_ch, 3, (.1, .5))
l_abs_x = unicorn.abs_sobel_thresh(l_ch, 'x', 5, (70, 250))
l_abs_y = unicorn.abs_sobel_thresh(l_ch, 'y', 5, (70, 250))
sobel_combo_l = np.zeros_like(l_dir_mag)
sobel_combo_l[((l_abs_x == 1) & (l_abs_y == 1)) | \
               ((l_mag == 1) & (l_dir_mag == 1))] = 1

# S channel processing
s_mag = unicorn.mag_thresh(s_ch, 5, (30, 40))
s_dir_mag = unicorn.dir_threshold(s_ch, 3, (.2, 1.5))
s_abs_x = unicorn.abs_sobel_thresh(s_ch, 'x', 5, (50, 150))
s_abs_y = unicorn.abs_sobel_thresh(s_ch, 'y', 5, (10, 150))
sobel_combo_s = np.zeros_like(s_dir_mag)
sobel_combo_s[((s_abs_x == 1) & (s_abs_y == 1)) | \
               ((s_mag == 1) & (s_dir_mag == 1))] = 1

print(sobel_combo_s)
print(sobel_combo_l)
# sobel_combo_l = np.array(sobel_combo_l).astype(np.int)
sobel_ls_or = cv2.bitwise_or(sobel_combo_l, sobel_combo_s)

sobel_ls_and = cv2.bitwise_and(sobel_combo_l, sobel_combo_s)

cv2.imshow('sobel_ls_or', sobel_ls_or)
# cv2.imshow('sobel and', sobel_ls_and)

# print(color_mask)
# print(np.array(sobel_ls_or).astype(np.int).shape, type(np.array(sobel_ls_or).astype(np.int)))
# print(np.array(sobel_ls_or).astype(np.int))
# # cv2.imshow('color', color_mask)
#
#
# print(color_mask.shape, type(color_mask))
# print(sobel_ls_or.shape, type(sobel_ls_or))
color_mask = np.array(color_mask).astype(np.float)
cv2.imshow('color', color_mask)
sobel_color_mask_or = cv2.bitwise_or(sobel_ls_or, color_mask)
sobel_color_mask_and = cv2.bitwise_and(sobel_ls_or, color_mask)

sobel_color_and_or = cv2.bitwise_or(sobel_ls_and, color_mask)

cv2.imshow('soble + color OR', sobel_color_mask_or)
cv2.imshow('sobel+color AND', sobel_color_mask_and)
cv2.imshow('sobel and + color OR', sobel_color_and_or)
#
# cv2.imshow('sobel ls and', sobel_ls_and)
# cv2.imshow('sobel ls', sobel_ls_or)
#
# cv2.imshow('sobel_combo_l', sobel_combo_l)
# cv2.imshow('sobel_combo_s', sobel_combo_s)



k = cv2.waitKey(0) & 0xFF
if k == 27: # k == ESC
    cv2.destroyAllWindows()
elif k == ord('s'): # k == s -> save and exit
    cv2.imwrite('output_images/warped1_1.png', warped)
    cv2.imwrite('output_images/color_mask1_1.png', color_mask)
    # cv2.imwrite('output_images/canny1_1.png', canny_img)
    # cv2.imwrite('output_images/Hough1_1.png', line_image)
    cv2.imwrite('output_images/adapt_threshold1_1.png', th2)
    cv2.imwrite('output_images/sobel_l1_1.png', sobel_combo_l)
    cv2.imwrite('output_images/sobel_s1_1.png', sobel_combo_s)
    cv2.imwrite('output_images/sobel_or1_1.png', sobel_ls_or)
    cv2.imwrite('output_images/sobel_and1_1.png', sobel_ls_and)

    cv2.destroyAllWindows()
