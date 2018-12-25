
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

# print(sobel_combo_s)
# print(sobel_combo_l)
cv2.imshow('sobel l', sobel_combo_l)
cv2.imshow('sobel_s', sobel_combo_s)

# sobel_combo_l = np.array(sobel_combo_l).astype(np.int)
sobel_ls_or = cv2.bitwise_or(sobel_combo_l, sobel_combo_s)

sobel_ls_and = cv2.bitwise_and(sobel_combo_l, sobel_combo_s)

cv2.imshow('sobel_ls_or', sobel_ls_or)
cv2.imshow('sobel and', sobel_ls_and)

# test_img = np.copy(sobel_ls_or)/255.
#
# test = test_img.astype(np.uint8)
# out_img = cv2.drawKeypoints(test, test_img, (255, 0, 0), 4)
#
# cv2.imshow('test', out_img)
#
#
# sobel_ls_or = np.array(sobel_ls_or).astype(np.int)
# bgr_img = cv2.cvtColor(sobel_ls_or, cv2.COLOR_GRAY2BGR)
# cv2.imshow('bgr', bgr_img)

canny_sobel = cv2.Canny(sobel_ls_or, 100, 200)
# cv2.imshow('canny', canny_sobel)

# print(color_mask)
# print(np.array(sobel_ls_or).astype(np.int).shape, type(np.array(sobel_ls_or).astype(np.int)))
# print(np.array(sobel_ls_or).astype(np.int))
# # cv2.imshow('color', color_mask)
#
#
# print(color_mask.shape, type(color_mask))
# print(sobel_ls_or.shape, type(sobel_ls_or))
color_mask = np.array(color_mask).astype(np.float)
# cv2.imshow('color', color_mask)
sobel_color_mask_or = cv2.bitwise_or(sobel_ls_or, color_mask)
sobel_color_mask_and = cv2.bitwise_and(sobel_ls_or, color_mask)

sobel_color_and_or = cv2.bitwise_or(sobel_ls_and, color_mask)

# gauss_com_img = cv2.GaussianBlur(sobel_color_and_or, (1, 1), 0)
img_size = warped.shape

half_comb_img = sobel_color_and_or[img_size[0]/2 : img_size[0], :]
half_comb_img = np.array(half_comb_img).astype(np.int)
# print(comb_img)
# mov_filtsize = img_size[1]/50.
# # print(mov_filtsize)
# mean_lane = np.mean(half_comb_img, axis=0)
# mean_lane = unicorn.moving_average(mean_lane, mov_filtsize)
#
# # print('mean lane', mean_lane)
# plt.plot(mean_lane>.05)
# plt.plot(mean_lane)
# plt.xlabel('image x')
# plt.ylabel('mean intensity')
# plt.show()
# print(img_size)
# print(len(mean_lane))
# arg_fsb = np.argwhere(mean_lane)
# print(arg_fsb)
#
# arg_fsb_L = arg_fsb[arg_fsb<img_size[1]/2]
# print(arg_fsb_L)
# arg_fsb_L_min = np.min(arg_fsb_L)
# arg_fsb_L_max = np.max(arg_fsb_L)
#
# img_L = np.copy(half_comb_img)
# img_L[:,0:arg_fsb_L_min] = 0
# img_L[:,arg_fsb_L_max:img_size[1]] = 0
#
#
# arg_fsb_R = arg_fsb[arg_fsb>img_size[1]/2.]
# arg_fsb_R_min = np.min(arg_fsb_R)
# arg_fsb_R_max = np.max(arg_fsb_R)
#
# img_R = np.copy(half_comb_img)
# img_R[:,0:arg_fsb_R_min] = 0
# img_R[:,arg_fsb_R_max:img_size[1]] = 0
# #img_L = gaussian_blur(img_L,5)
#
#
#
# plt.subplot(1,2,1)
# plt.imshow(img_L,cmap='gray')
# plt.axis('off')
# plt.title('Left lane markings')
# plt.subplot(1,2,2)
# plt.imshow(img_R,cmap='gray')
# plt.axis('off')
# plt.title('Right lane markings')
# plt.show()
# cv2.imshow('soble + color OR', sobel_color_mask_or)
# cv2.imshow('sobel+color AND', sobel_color_mask_and)
img_y2 = img_size[0]
img_y1 = img_size[0]-img_size[0]/8
lane1 = sobel_color_and_or[img_y1:img_y2, :]
cv2.imshow('lane1', lane1)


# for i in range(8):
#     img_y1 = img_size[0]-img_size[0]*i/8
#     img_y2 = img_size[0]-img_size[0]*(i+1)/8
#     print(img_y1, img_y2)
#     plt.subplot(8,1,8-i)
#     plt.imshow(half_comb_img[img_y2:img_y1,:], cmap='gray')
#     plt.axis('off')
#
#     plt.show()

cv2.imshow('sobel and + color OR', sobel_color_and_or)




k = cv2.waitKey(0) & 0xFF
if k == 27: # k == ESC
    cv2.destroyAllWindows()
elif k == ord('s'): # k == s -> save and exit
    cv2.imwrite('output_images/warped1_1.png', warped)
    cv2.imwrite('output_images/color_mask1_1.png', color_mask)
    # cv2.imwrite('output_images/canny1_1.png', canny_img)
    # cv2.imwrite('output_images/Hough1_1.png', line_image)
    cv2.imwrite('output_images/adapt_threshold1_1.png', th2)


    cv2.destroyAllWindows()
