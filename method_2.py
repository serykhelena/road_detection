#!/usr/bin/python


# from __future__ import print_function
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import math
import pickle
import unicorn
from scipy import ndimage
import scipy
# from skimage.filters import roberts, sobel
'''
with open('cal_param_opts.txt', 'rb') as fp:
    opts = pickle.load(fp)

with open('cal_param_ipts.txt', 'rb') as fp:
    ipts = pickle.load(fp)
'''

pic_num = '000'


image = cv2.imread('input_masks/frame000' + pic_num + '_mask.pgm')
# undist_img = unicorn.undistort_image(image, opts, ipts)

# warped = unicorn.bird_eye_view(undist_img)   # (maxWidth, maxHeight))

crop_img = unicorn.find_low_border_of_roi_m2(image, roi_window=10, x_limit=60, show_border=0)

height = crop_img.shape[0]
width = crop_img.shape[1]

lb_pnt, rb_pnt, lt_pnt, rt_pnt = unicorn.get_4_pnts_for_warping(crop_img)

quad_img = cv2.resize(crop_img, (width, 100))
warped_img = unicorn.bird_eye_view(quad_img, lt_pnt, rt_pnt, rb_pnt, lb_pnt)

gray_img = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)

mean_lines, mean_indexes = unicorn.get_mean_intensity(gray_img)
boundaries = unicorn.get_boundaries_for_intensity(mean_lines, 5)

extrema_count = unicorn.count_number_of_extrema(mean_lines)

first_peak, second_peak, third_peak, first_x, second_x, third_x = unicorn.get_picks_arrays(extrema_count, mean_lines, boundaries, mean_indexes)

first_pick_coo, second_pick_coo, third_pick_coo = unicorn.get_picks_coordinates(gray_img, first_peak, first_x, second_peak, second_x, third_peak, third_x)

# print("Left pick", first_pick_coo)
# print("Center pick", second_pick_coo)
# print("Right pick", third_pick_coo)
#
#
# cv2.circle(warped_img, (first_pick_coo[0], first_pick_coo[1]), 3, (255, 0, 0), thickness=-1)
# cv2.circle(warped_img, (second_pick_coo[0], second_pick_coo[1]), 3, (0, 255, 0), thickness=-1)
# cv2.circle(warped_img, (third_pick_coo[0], third_pick_coo[1]), 3, (0, 0, 255), thickness=-1)


warp_height = warped_img.shape[0]
warp_width = warped_img.shape[1]

w_num = 5
dh = warp_height / w_num

warp1 = warped_img[0:dh, :]
mean_1, indx_1 = unicorn.get_mean_intensity(warp1)
unicorn.process_part_of_img(warp1, 5, 20)
# mean_1, ind_1 = unicorn.get_mean_intensity(warp1)
# bound_1 = unicorn.get_boundaries_for_intensity(mean_1, 5)
# extr_1 = unicorn.count_number_of_extrema(mean_1)
# f_pick_1, s_pick_1, t_pick_1, f_x_1, s_x_1, t_x_1 = unicorn.get_picks_arrays(extr_1, mean_1, bound_1, ind_1)
# f_pick_coo_1, s_pick_coo_1, t_pick_coo_1 = unicorn.get_picks_coordinates(warp1, f_pick_1, f_x_1, s_pick_1, s_x_1, t_pick_1, t_x_1)
#
# print("Left 1", f_pick_coo_1)
# unicorn.draw_pick_pnts(warp1, f_pick_coo_1, s_pick_coo_1, t_pick_coo_1)
# unicorn.draw_pick_mask(warp1, 20, f_pick_coo_1)
# unicorn.draw_pick_mask(warp1, 20, s_pick_coo_1)
# unicorn.draw_pick_mask(warp1, 20, t_pick_coo_1)



warp2 = warped_img[dh:2*dh, :]
mean_2, ind_2 = unicorn.get_mean_intensity(warp2)
warp3 = warped_img[2*dh:3*dh, :]
mean_3, ind_3 = unicorn.get_mean_intensity(warp3)
warp4 = warped_img[3*dh:4*dh, :]
mean_4, ind_4 = unicorn.get_mean_intensity(warp4)
warp5 = warped_img[4*dh:5*dh, :]
mean_5, ind_5 = unicorn.get_mean_intensity(warp5)


fig = plt.figure(figsize=(8, 8))

plt.subplot(3,1,1)
plt.imshow(gray_img,cmap='gray')
plt.axis('off')
plt.subplot(3,1,2)
plt.imshow(warped_img)
plt.subplot(3,1,3)
plt.plot(boundaries)
plt.plot(mean_lines)
# plt.plot(first_peak)
# plt.plot(second_peak)
# plt.plot(third_peak)
plt.xlabel('image x')
plt.ylabel('mean intensity')
plt.xlim(0, width)

fig1 = plt.figure(figsize=(8, 8))
plt.subplot(10,1,1)
plt.imshow(warp1)
plt.subplot(10,1,2)
plt.plot(mean_1)
plt.xlim(0, width)
plt.subplot(10,1,3)
plt.imshow(warp2)
plt.subplot(10,1,4)
plt.plot(mean_2)
plt.xlim(0, width)
plt.subplot(10,1,5)
plt.imshow(warp3)
plt.subplot(10,1,6)
plt.plot(mean_3)
plt.xlim(0, width)
plt.subplot(10,1,7)
plt.imshow(warp4)
plt.subplot(10,1,8)
plt.plot(mean_4)
plt.xlim(0, width)
plt.subplot(10,1,9)
plt.imshow(warp5)
plt.subplot(10,1,10)
plt.plot(mean_5)
plt.xlim(0, width)

# fig.add_subplot(3, 1, 1)
# plt.imshow(crop_img)
# fig.add_subplot(2, 1, 1)
# plt.imshow(warped_img)
# fig.add_subplot(2, 1, 2)
# plt.imshow(canny_img)
# plt.savefig('hist_imgs/hist2000' + pic_num + '.png')

plt.show()




cv2.imwrite('warped_imgs/warp000' + pic_num + '.png', warped_img)
cv2.imwrite('roi_imgs/roi2000' + pic_num + '.png', crop_img)
# k = cv2.waitKey(0) & 0xFF
# if k == 27: # k == ESC
#     cv2.destroyAllWindows()
# elif k == ord('s'): # k == s -> save and exit
#     # cv2.imwrite('output_images/warped1_1.png', warped)
#     # cv2.imwrite('output_images/color_mask1_1.png', color_mask)
#     # cv2.imwrite('output_images/canny1_1.png', canny_img)
#     # cv2.imwrite('output_images/Hough1_1.png', line_image)
#     # cv2.imwrite('output_images/adapt_threshold1_1.png', th2)
#
#     cv2.destroyAllWindows()
