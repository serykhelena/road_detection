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

first_peak, second_peak, third_peak, first_x, second_x, third_x = unicorn.get_picks_arrays(extrema_count, mean_lines, boundaries, mean_indexes, gray_img)

first_pick_coo, second_pick_coo, third_pick_coo = unicorn.get_picks_coordinates(gray_img, first_peak, first_x, second_peak, second_x, third_peak, third_x)


warp_height = warped_img.shape[0]
warp_width = warped_img.shape[1]

w_num = 5
dh = warp_height / w_num

warp1 = np.copy(warped_img[0:dh, :])
mean_1, indx_1 = unicorn.get_mean_intensity(warp1)
# unicorn.draw_magic(warp1, 5, 20)
extrema_1 = unicorn.count_number_of_extrema(mean_1)
# print("NUMBER PICKS 1", extrema_1)
left_mask_1, center_mask_1, right_mask_1 = unicorn.get_mask_data(warp1, 5, 20)


warp2 = np.copy(warped_img[dh:2*dh, :])
mean_2, ind_2 = unicorn.get_mean_intensity(warp2)
# unicorn.draw_magic(warp2, 5, 20)
left_mask_2, center_mask_2, right_mask_2 = unicorn.get_mask_data(warp2, 5, 20)


warp3 = np.copy(warped_img[2*dh:3*dh, :])
mean_3, ind_3 = unicorn.get_mean_intensity(warp3)
# unicorn.draw_magic(warp3, 5, 20)
left_mask_3, center_mask_3, right_mask_3 = unicorn.get_mask_data(warp3, 5, 20)


warp4 = np.copy(warped_img[3*dh:4*dh, :])
mean_4, ind_4 = unicorn.get_mean_intensity(warp4)
# unicorn.draw_magic(warp4, 5, 20)
left_mask_4, center_mask_4, right_mask_4 = unicorn.get_mask_data(warp4, 5, 20)


warp5 = np.copy(warped_img[4*dh:5*dh, :])
mean_5, ind_5 = unicorn.get_mean_intensity(warp5)
# unicorn.draw_magic(warp5, 5, 20)
left_mask_5, center_mask_5, right_mask_5 = unicorn.get_mask_data(warp5, 5, 20)

left_mask = np.copy(warped_img)
for y in range(0, left_mask.shape[0]):
    for x in range(0, left_mask.shape[1]):
        left_mask[y][x] = [0, 0, 0]

center_mask = np.copy(warped_img)
for y in range(0, center_mask.shape[0]):
    for x in range(0, center_mask.shape[1]):
        center_mask[y][x] = [0, 0, 0]

right_mask = np.copy(warped_img)
for y in range(0, right_mask.shape[0]):
    for x in range(0, right_mask.shape[1]):
        right_mask[y][x] = [0, 0, 0]

# y -> 0 - 20
for y in range(0, warp1.shape[0]):
    for x in range(0, warp1.shape[1]):
        left_mask[y][x] = left_mask_1[y][x]
        center_mask[y][x] = center_mask_1[y][x]
        right_mask[y][x] = right_mask_1[y][x]

# y -> 20 - 40
for y in range(0, warp1.shape[0]):
    for x in range(0, warp1.shape[1]):
        left_mask[y+20][x] = left_mask_2[y][x]
        center_mask[y+20][x] = center_mask_2[y][x]
        right_mask[y+20][x] = right_mask_2[y][x]

# y -> 40 - 60
for y in range(0, warp1.shape[0]):
    for x in range(0, warp1.shape[1]):
        left_mask[y + 40][x] = left_mask_3[y][x]
        center_mask[y + 40][x] = center_mask_3[y][x]
        right_mask[y + 40][x] = right_mask_3[y][x]

# y -> 60 - 80
for y in range(0, warp1.shape[0]):
    for x in range(0, warp1.shape[1]):
        left_mask[y + 60][x] = left_mask_4[y][x]
        center_mask[y + 60][x] = center_mask_4[y][x]
        right_mask[y + 60][x] = right_mask_4[y][x]

# y -> 80 - 100
for y in range(0, warp1.shape[0]):
    for x in range(0, warp1.shape[1]):
        left_mask[y + 80][x] = left_mask_5[y][x]
        center_mask[y + 80][x] = center_mask_5[y][x]
        right_mask[y + 80][x] = right_mask_5[y][x]

filt_left = np.copy(warped_img)
filt_left = cv2.bitwise_and(warped_img, left_mask)

filt_center = np.copy(warped_img)
filt_center = cv2.bitwise_and(warped_img, center_mask)

filt_right = np.copy(warped_img)
filt_right = cv2.bitwise_and(warped_img, right_mask)

filt_all = np.copy(warped_img)
filt_all = cv2.bitwise_or(filt_left, filt_center)
filt_all = cv2.bitwise_or(filt_all, filt_right)


vals = np.argwhere(filt_left>200)
filt_left_y = vals.T[0]
filt_left_x = vals.T[1]


left_fit = np.polyfit(filt_left_x, filt_left_y, 1)
left_fit_y = np.arange(11)*filt_left.shape[0]/10
left_fit_y[len(left_fit_y)-1] = filt_left.shape[0] - 1
left_fit_x = (left_fit_y - left_fit[1]) / left_fit[0]
# left_fit_x = left_fit[0]*left_fit_y**2 + left_fit[1]*left_fit_y + left_fit[2]


vals = np.argwhere(filt_center>200)
filt_center_y = vals.T[0]
filt_center_x = vals.T[1]

center_fit = np.polyfit(filt_center_x, filt_center_y, 1)
center_fit_y = np.arange(11)*filt_center.shape[0]/10
center_fit_y[len(center_fit_y)-1] = filt_center.shape[0] - 1
center_fit_x = (center_fit_y - center_fit[1]) / center_fit[0]

vals = np.argwhere(filt_right>250)
filt_right_y = vals.T[0]
filt_right_x = vals.T[1]

right_fit = np.polyfit(filt_right_x, filt_right_y, 1)

right_fit_y = np.arange(11)*filt_right.shape[0]/10
right_fit_y[len(right_fit_y)-1] = filt_right.shape[0] - 1
right_fit_x = (right_fit_y - right_fit[1]) / right_fit[0]








fig6 = plt.figure(figsize=(8, 8))
# plt.subplot(4,1,1)
plt.imshow(filt_all)
plt.plot(left_fit_x, left_fit_y, 'red', linewidth=5)
plt.plot(center_fit_x, center_fit_y, 'blue', linewidth=5)
plt.plot(right_fit_x, right_fit_y, 'green', linewidth=5)


# fig4 = plt.figure(figsize=(8, 8))
#
# plt.subplot(4,1,1)
# plt.imshow(warped_img)
# plt.subplot(4,1,2)
# plt.imshow(left_mask)
# # plt.title('left mask')
# plt.subplot(4,1,3)
# plt.imshow(center_mask)
# # plt.title('center mask')
# plt.subplot(4,1,4)
# plt.imshow(right_mask)
# # plt.title('right mask')
#
# fig5 = plt.figure(figsize=(8, 8))
# plt.subplot(4,1,1)
# plt.imshow(warped_img)
# plt.subplot(4,1,2)
# plt.imshow(filt_left)
#
# plt.subplot(4,1,3)
# plt.imshow(filt_center)
#
# plt.subplot(4,1,4)
# plt.imshow(filt_right)


# fig = plt.figure(figsize=(8, 8))

# # plt.subplot(3,1,1)
# # plt.imshow(test_left_mask)
# # plt.axis('off')
# plt.subplot(3,1,2)
# plt.imshow(warped_img)
# plt.subplot(3,1,3)
# plt.plot(boundaries)
# plt.plot(mean_lines)
# # plt.plot(first_peak)
# # plt.plot(second_peak)
# # plt.plot(third_peak)
# plt.xlabel('image x')
# plt.ylabel('mean intensity')
# plt.xlim(0, width)

# fig1 = plt.figure(figsize=(8, 8))
# plt.subplot(10,1,1)
# plt.imshow(warp1)
# plt.subplot(10,1,2)
# plt.plot(mean_1)
# plt.xlim(0, width)
# plt.subplot(10,1,3)
# plt.imshow(warp2)
# plt.subplot(10,1,4)
# plt.plot(mean_2)
# plt.xlim(0, width)
# plt.subplot(10,1,5)
# plt.imshow(warp3)
# plt.subplot(10,1,6)
# plt.plot(mean_3)
# plt.xlim(0, width)
# plt.subplot(10,1,7)
# plt.imshow(warp4)
# plt.subplot(10,1,8)
# plt.plot(mean_4)
# plt.xlim(0, width)
# plt.subplot(10,1,9)
# plt.imshow(warp5)
# plt.subplot(10,1,10)
# plt.plot(mean_5)
# plt.xlim(0, width)

# fig2 = plt.figure(figsize=(8, 8))
# plt.subplot(5,1,1)
# plt.imshow(warp4)
# plt.subplot(5,1,2)
# plt.plot(mean_4)
# plt.xlim(0, width)
# plt.subplot(5,1,3)
# plt.imshow(left_mask_4)
# plt.title('left mask')
# plt.subplot(5,1,4)
# plt.imshow(center_mask_4)
# plt.title('center mask')
# plt.subplot(5,1,5)
# plt.imshow(right_mask_4)
# plt.title('right mask')
#
# fig3 = plt.figure(figsize=(8, 8))
# plt.subplot(5,1,1)
# plt.imshow(warp2)
# plt.subplot(5,1,2)
# plt.plot(mean_2)
# plt.xlim(0, width)
# plt.subplot(5,1,3)
# plt.imshow(left_mask_2)
# plt.title('left mask')
# plt.subplot(5,1,4)
# plt.imshow(center_mask_2)
# plt.title('center mask')
# plt.subplot(5,1,5)
# plt.imshow(right_mask_2)
# plt.title('right mask')


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
