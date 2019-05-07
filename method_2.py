#!/usr/bin/python

import numpy as np
import cv2
import unicorn
import time
import matplotlib.pyplot as plt

start = time.time()
pic_num = '256'

ref_img = cv2.imread('input_imgs/frame000' + pic_num + '.png')
image = cv2.imread('input_masks/frame000' + pic_num + '_mask.pgm')

crop_img = unicorn.find_low_border_of_roi_m2(image, y_limit=10, x_limit=60, show_border=0)
crop_img_lined = unicorn.find_low_border_of_roi_m2(image, y_limit=10, x_limit=60, show_border=1)
height = crop_img.shape[0]
width = crop_img.shape[1]

fig0 = plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.imshow(crop_img)
plt.subplot(2, 1, 2)
plt.imshow(crop_img_lined)
plt.show()

# dx 000 = 2
# dx 007 = 20
# dx 090 = 10
# dx 118 = 5
# dx 170 = 2
# dx 235 = 0
# dx 256 = 0


lb_pnt, rb_pnt, lt_pnt, rt_pnt = unicorn.get_4_pnts_for_warping(crop_img, 0, draw_pnts=1)
warped_img = unicorn.bird_eye_view(crop_img, lt_pnt, rt_pnt, rb_pnt, lb_pnt)


# ''' to get image for article (fig 9)
fig = plt.figure(figsize=(8, 8))
plt.subplot(2,1,1)
plt.imshow(crop_img)
plt.subplot(2,1,2)
plt.imshow(warped_img)
# '''
plt.show()

gray_img = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
mean_lines, mean_indexes = unicorn.get_mean_intensity(gray_img)
boundaries = unicorn.get_boundaries_for_intensity(mean_lines, 5)

''' to get image for article (fig 10)
fig1 = plt.figure(figsize=(8, 4))
plt.subplot(2,1,1)
plt.imshow(warped_img)
plt.subplot(2,1,2)
plt.plot(boundaries)
plt.plot(mean_lines)
plt.xlim(0, width)
plt.ylim(0, height)
plt.ylabel('mean intensity')
plt.xlabel('x')
'''

extrema_count = unicorn.count_number_of_extrema(mean_lines)

first_peak, second_peak, third_peak, first_x, second_x, third_x = unicorn.get_picks_arrays(extrema_count, mean_lines, boundaries, mean_indexes, gray_img)
first_pick_coo, second_pick_coo, third_pick_coo = unicorn.get_picks_coordinates(gray_img, first_peak, first_x, second_peak, second_x, third_peak, third_x)

warp_height = warped_img.shape[0]
warp_width = warped_img.shape[1]

w_num = 5
dh = warp_height / w_num
int_dh = w_num * dh
last_dh = warp_height - int_dh

line_width = 6
bound_limit = 0.2

warp1 = np.copy(warped_img[0:dh, :])
mean_1, indx_1 = unicorn.get_mean_intensity(warp1)
# unicorn.draw_magic(warp1, bound_limit, line_width)
left_mask_1, center_mask_1, right_mask_1 = unicorn.get_mask_data(warp1, bound_limit, line_width)

warp2 = np.copy(warped_img[dh:2*dh, :])
mean_2, ind_2 = unicorn.get_mean_intensity(warp2)
# unicorn.draw_magic(warp2, bound_limit, line_width)
left_mask_2, center_mask_2, right_mask_2 = unicorn.get_mask_data(warp2, bound_limit, line_width)

warp3 = np.copy(warped_img[2*dh:3*dh, :])
mean_3, ind_3 = unicorn.get_mean_intensity(warp3)
# unicorn.draw_magic(warp3, bound_limit, line_width)
left_mask_3, center_mask_3, right_mask_3 = unicorn.get_mask_data(warp3, bound_limit, line_width)

warp4 = np.copy(warped_img[3*dh:4*dh, :])
mean_4, ind_4 = unicorn.get_mean_intensity(warp4)
# unicorn.draw_magic(warp4, bound_limit, line_width)
left_mask_4, center_mask_4, right_mask_4 = unicorn.get_mask_data(warp4, bound_limit, line_width)

warp5 = np.copy(warped_img[4*dh:(5*dh + last_dh), :])
mean_5, ind_5 = unicorn.get_mean_intensity(warp5)
# unicorn.draw_magic(warp5, bound_limit, line_width)
left_mask_5, center_mask_5, right_mask_5 = unicorn.get_mask_data(warp5, bound_limit, line_width)

# ''' to get pic for article (fig 12)
fig2 = plt.figure(figsize=(8, 8))
plt.subplot(10,1,1)
plt.imshow(warp1)
plt.subplot(10,1,2)
plt.plot(mean_1)
plt.xlim(0, width)
plt.ylim(0, height)
plt.subplot(10,1,3)
plt.imshow(warp2)
plt.subplot(10,1,4)
plt.plot(mean_2)
plt.xlim(0, width)
plt.ylim(0, height)
plt.subplot(10,1,5)
plt.imshow(warp3)
plt.subplot(10,1,6)
plt.plot(mean_3)
plt.xlim(0, width)
plt.ylim(0, height)
plt.subplot(10,1,7)
plt.imshow(warp4)
plt.subplot(10,1,8)
plt.plot(mean_4)
plt.xlim(0, width)
plt.ylim(0, height)
plt.subplot(10,1,9)
plt.imshow(warp5)
plt.subplot(10,1,10)
plt.plot(mean_5)
plt.xlim(0, width)
plt.ylim(0, height)


fig3 = plt.figure(figsize=(8, 4))
plt.imshow(warped_img)
# '''
plt.show()

left_mask = unicorn.make_image_all_black(warped_img)
center_mask = unicorn.make_image_all_black(warped_img)
right_mask = unicorn.make_image_all_black(warped_img)



# y -> 0 - 12
for y in range(0, warp1.shape[0]):
    for x in range(0, warp1.shape[1]):
        left_mask[y][x] = left_mask_1[y][x]
        center_mask[y][x] = center_mask_1[y][x]
        right_mask[y][x] = right_mask_1[y][x]

# y -> 12 - 24
for y in range(0, warp1.shape[0]):
    for x in range(0, warp1.shape[1]):
        left_mask[y+warp1.shape[0]][x] = left_mask_2[y][x]
        center_mask[y+warp1.shape[0]][x] = center_mask_2[y][x]
        right_mask[y+warp1.shape[0]][x] = right_mask_2[y][x]

# y -> 24 - 36
for y in range(0, warp1.shape[0]):
    for x in range(0, warp1.shape[1]):
        left_mask[y + 2*warp1.shape[0]][x] = left_mask_3[y][x]
        center_mask[y + 2*warp1.shape[0]][x] = center_mask_3[y][x]
        right_mask[y + 2*warp1.shape[0]][x] = right_mask_3[y][x]

# y -> 36 - 48
for y in range(0, warp1.shape[0]):
    for x in range(0, warp1.shape[1]):
        left_mask[y + 3*warp1.shape[0]][x] = left_mask_4[y][x]
        center_mask[y + 3*warp1.shape[0]][x] = center_mask_4[y][x]
        right_mask[y + 3*warp1.shape[0]][x] = right_mask_4[y][x]

# y -> 48 - 60
for y in range(0, warp1.shape[0]):
    for x in range(0, warp1.shape[1]):
        left_mask[y + 4*warp1.shape[0]][x] = left_mask_5[y][x]
        center_mask[y + 4*warp1.shape[0]][x] = center_mask_5[y][x]
        right_mask[y + 4*warp1.shape[0]][x] = right_mask_5[y][x]

# y -> 60 - 62
for y in range(0, last_dh):
    for x in range(0, warp1.shape[1]):
        left_mask[y + 5*warp1.shape[0]][x] = left_mask_5[y][x]
        center_mask[y + 5*warp1.shape[0]][x] = center_mask_5[y][x]
        right_mask[y + 5*warp1.shape[0]][x] = right_mask_5[y][x]

''' fig 14
fig4 = plt.figure(figsize=(8, 8))
plt.subplot(5,1,1)
plt.imshow(warp4)
plt.subplot(5,1,2)
plt.plot(mean_4)
plt.xlim(0, width)
plt.ylim(0, warp4.shape[0])
plt.subplot(5,1,3)
plt.title('left mask')
plt.imshow(left_mask_4)
plt.subplot(5,1,4)
plt.title('center mask')
plt.imshow(center_mask_4)
plt.subplot(5,1,5)
plt.title('right mask')
plt.imshow(right_mask_4)
'''

''' fig 15
fig5 = plt.figure(figsize=(8, 8))
plt.subplot(4,1,1)
plt.imshow(warped_img)
plt.subplot(4,1,2)
plt.title('left mask')
plt.imshow(left_mask)
plt.subplot(4,1,3)
plt.title('center mask')
plt.imshow(center_mask)
plt.subplot(4,1,4)
plt.title('right mask')
plt.imshow(right_mask)
'''

filt_left = np.copy(warped_img)
filt_left = cv2.bitwise_and(warped_img, left_mask)

filt_center = np.copy(warped_img)
filt_center = cv2.bitwise_and(warped_img, center_mask)

filt_right = np.copy(warped_img)
filt_right = cv2.bitwise_and(warped_img, right_mask)

filt_all = np.copy(warped_img)
filt_all = cv2.bitwise_or(filt_left, filt_center)
filt_all = cv2.bitwise_or(filt_all, filt_right)

''' fig 16
fig6 = plt.figure(figsize=(8, 8))
plt.subplot(4,1,1)
plt.imshow(filt_all)
plt.subplot(4,1,2)
plt.title('left mask')
plt.imshow(filt_left)
plt.subplot(4,1,3)
plt.title('center mask')
plt.imshow(filt_center)
plt.subplot(4,1,4)
plt.title('right mask')
plt.imshow(filt_right)
'''

vals = np.argwhere(filt_left>85)
filt_left_y = vals.T[0]
filt_left_x = vals.T[1]

temp_high_x = []
for x in range(0, filt_left.shape[1]):
    if filt_left[0][x][0] > 0:
        temp_high_x.append(x)

high_x = int(np.mean(temp_high_x))

temp_low_x = []
for x in range(0, filt_left.shape[1]):
    if filt_left[filt_left.shape[0]-1][x][0] > 0:
        temp_low_x.append(x)

low_x = int(np.mean(temp_low_x))

left_fit_x = np.array([high_x, low_x])
left_fit_y = np.array([0, filt_left.shape[0]-1])

temp_high_x[:] = []
temp_low_x[:] = []

# left_fit = np.polyfit(filt_left_x, filt_left_y, 2)
# left_fit_y = np.array([0, filt_left.shape[0]-1])
# left_fit_y = np.arange(11)*filt_left.shape[0]/10
# left_fit_y[len(left_fit_y)-1] = filt_left.shape[0] - 1
# print 'left fit y', left_fit_y
# left_fit_x = (left_fit_y - round(left_fit[1], 4)) / round(left_fit[0], 4)
# left_fit_x = left_fit[0]*left_fit_y**2 + left_fit[1]*left_fit_y + left_fit[2]
# left_fit_x = np.array(left_fit_x).astype(int)
# print 'left fit x', left_fit_x


vals = np.argwhere(filt_center>200)
filt_center_y = vals.T[0]
filt_center_x = vals.T[1]

min_center_y = min(filt_center_y)

temp_high_x = []
for x in range(0, filt_center.shape[1]):
    if filt_center[min_center_y][x][0] > 0:
        temp_high_x.append(x)

high_x = int(np.mean(temp_high_x))

temp_low_x = []
for x in range(0, filt_center.shape[1]):
    if filt_center[filt_center.shape[0]-1][x][0] > 0:
        temp_low_x.append(x)

low_x = int(np.mean(temp_low_x))

center_fit_x = np.array([high_x, low_x])
center_fit_y = np.array([0, filt_center.shape[0]-1])

temp_high_x[:] = []
temp_low_x[:] = []

# center_fit = np.polyfit(filt_center_x, filt_center_y, 1)
# center_fit_y = np.arange(11)*filt_center.shape[0]/10
# center_fit_y[len(center_fit_y)-1] = filt_center.shape[0] - 1
# center_fit_x = (center_fit_y - round(center_fit[1], 4)) / round(center_fit[0], 4)
# center_fit_x = center_fit[0]*center_fit_y**2 + center_fit[1]*center_fit_y + center_fit[2]


vals = np.argwhere(filt_right>39)
filt_right_y = vals.T[0]
filt_right_x = vals.T[1]

min_right_y = min(filt_right_y)

temp_high_x = []
for x in range(0, filt_right.shape[1]):
    if filt_right[min_right_y][x][0] > 0:
        temp_high_x.append(x)

high_x = int(np.mean(temp_high_x))

temp_low_x = []
for x in range(0, filt_right.shape[1]):
    if filt_right[filt_right.shape[0]-1][x][0] > 0:
        temp_low_x.append(x)

low_x = int(np.mean(temp_low_x))

right_fit_x = np.array([high_x, low_x])
right_fit_y = np.array([0, filt_center.shape[0]-1])

temp_high_x[:] = []
temp_low_x[:] = []

# right_fit = np.polyfit(filt_right_x, filt_right_y, 2)
# right_fit_y = np.arange(4)*filt_right.shape[0]/3
# right_fit_y[len(right_fit_y)-1] = filt_right.shape[0] - 1
# right_fit_x = (right_fit_y - round(right_fit[1], 4)) / round(right_fit[0], 4)
# right_fit_x = right_fit[0]*right_fit_y**2 + right_fit[1]*right_fit_y + right_fit[2]
# right_fit_x = abs(right_fit_x)
# print 'right y', right_fit_y
# print 'right x', right_fit_x

''' fig 17
fig7 = plt.figure(figsize=(8, 4))
plt.imshow(filt_all)
plt.plot(left_fit_x, left_fit_y, 'red', linewidth=5)
plt.plot(center_fit_x, center_fit_y, 'blue', linewidth=5)
plt.plot(right_fit_x, right_fit_y, 'green', linewidth=5)
'''

gray_filt_all = cv2.cvtColor(filt_all, cv2.COLOR_BGR2GRAY)
warp_zero = np.zeros_like(gray_filt_all).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

pnts_left_fit = np.array([np.transpose(np.vstack([left_fit_x, left_fit_y]))]).astype(np.int)
pnts_center_fit = np.array([np.flipud(np.transpose(np.vstack([center_fit_x, center_fit_y])))]).astype(np.int)
pnts_right_fit = np.array([np.transpose(np.vstack([right_fit_x, right_fit_y]))]).astype(np.int)

pnts_left_line = np.hstack((pnts_left_fit, pnts_center_fit))
pnts_right_line = np.hstack((pnts_center_fit, pnts_right_fit))
pnts_all_fit = np.hstack((pnts_left_fit, pnts_center_fit, pnts_right_fit))

cv2.fillPoly(filt_all, np.int_([pnts_left_line]), (100,225, 240))
cv2.fillPoly(filt_all, np.int_([pnts_right_line]), (245, 120, 220))

color_L = [240, 55, 100]
color_C = [40, 130, 210]
color_R = [70, 200, 100]


unicorn.draw_magic_line(filt_all, np.int_(pnts_left_fit), color_L)
unicorn.draw_magic_line(filt_all, np.int_(pnts_center_fit), color_C)
unicorn.draw_magic_line(filt_all, np.int_(pnts_right_fit), color_R)

# unicorn.draw_pw_lines(filt_all, np.int_(pnts_left_fit), color_L)
# unicorn.draw_pw_lines(filt_all, np.int_(pnts_center_fit), color_C)
# unicorn.draw_pw_lines(filt_all, np.int_(pnts_right_fit), color_R)

'''fig 18
fig8 = plt.figure(figsize=(8, 4))
plt.imshow(filt_all)
'''

color_warp = np.copy(filt_all)

small_col_warp = cv2.resize(color_warp, (width, height))

# lb_pnt, rb_pnt, lt_pnt, rt_pnt

lt_indx = np.where(left_fit_y == min(left_fit_y))
unw_lt_pnt = [int(left_fit_x[lt_indx]), min(left_fit_y)]

rt_indx = np.where(right_fit_y == min(right_fit_y))
unw_rt_pnt = [int(right_fit_x[rt_indx]), min(right_fit_y)]

lb_indx = np.where(left_fit_y == max(left_fit_y))
# unw_lb_pnt = [int(left_fit_x[lb_indx]), max(left_fit_y)]
unw_lb_pnt = [0, height]


rb_indx = np.where(right_fit_y == max(right_fit_y))
# unw_rb_pnt = [int(right_fit_x[rb_indx]), max(right_fit_y)]
unw_rb_pnt = [width, height]
# warped_img = unicorn.bird_eye_view(quad_img, lt_pnt, rt_pnt, rb_pnt, lb_pnt)

unwarp_img = unicorn.undo_bird_eye_view(small_col_warp,
                                        unw_lt_pnt, unw_rt_pnt, unw_rb_pnt, unw_lb_pnt,
                                        lt_pnt, rt_pnt, [width, height], [0, height])

ref_img = cv2.resize(ref_img, (image.shape[1], image.shape[0]))
ref_crop_img = ref_img[0:height, :]
res_img = cv2.addWeighted(ref_crop_img, 1, unwarp_img, 0.5, 0)



# unw_left_x = []
# unw_left_y = []
#
# color_L = np.array(color_L)
# for y in range(0, unwarp_img.shape[0]):
#     for x in range(0, unwarp_img.shape[1]):
#         # print type(unwarp_img[y][x]), unwarp_img[y][x], color_L, type(color_L)
#         if unwarp_img[y][x][0] == color_L[0] and unwarp_img[y][x][1] == color_L[1] and unwarp_img[y][x][2] == color_L[2]:
#             # print unwarp_img[y][x], color_L
#             unw_left_x.append(x)
#             unw_left_y.append(y)
#
# unw_left_y_min = min(unw_left_y)
# unw_lefT_y_max = max(unw_left_y)
#
# unw_temp_high_x = []
# unw_temp_low_x = []
# for x in range(0, unwarp_img.shape[1]):
#     if unwarp_img[unw_left_y_min][x][0] == color_L[0] and unwarp_img[unw_left_y_min][x][1] == color_L[1] and unwarp_img[unw_left_y_min][x][2] == color_L[2]:
#         unw_temp_high_x.append(x)
#
#     if unwarp_img[unw_lefT_y_max][x][0] == color_L[0] and unwarp_img[unw_lefT_y_max][x][1] == color_L[1] and unwarp_img[unw_lefT_y_max][x][2] == color_L[2]:
#         unw_temp_low_x.append(x)
#
# unw_left_high_x = int(np.mean(unw_temp_high_x))
# unw_left_low_x = int(np.mean(unw_temp_low_x))
#
# unw_left_line = [unw_left_high_x, unw_left_y_min, unw_left_low_x, unw_lefT_y_max]
unw_left_line = unicorn.get_pnts_of_line(unwarp_img, color_L)
unw_left_k, unw_left_b = unicorn.get_k_b_line(unw_left_line)
unw_left_k_deg = unicorn.get_k_deg(unw_left_line)
# print 'left k', unw_left_k, 'left_k_deg', unw_left_k_deg

unw_center_line = unicorn.get_pnts_of_line(unwarp_img, color_C)
unw_center_k_deg = unicorn.get_k_deg(unw_center_line)

unw_right_line = unicorn.get_pnts_of_line(unwarp_img, color_R)
unw_right_k_deg = unicorn.get_k_deg(unw_right_line)


# cv2.line(res_img, (unw_left_low_x, unw_lefT_y_max), (unw_left_high_x, unw_left_y_min), [0, 255, 0], thickness=5)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(unwarp_img, str(unw_left_k_deg), (5, 25), font, 1, (255, 0, 0), thickness=2)
cv2.putText(unwarp_img, str(unw_center_k_deg), (150, 25), font, 1, (255, 0, 0), thickness=2)
cv2.putText(unwarp_img, str(unw_right_k_deg), (250, 25), font, 1, (255, 0, 0), thickness=2)

end = time.time()
print(end - start)
''' fig 19
fig9= plt.figure(figsize=(8, 6))
plt.subplot(2, 1, 1)
plt.imshow(filt_all)
plt.subplot(2, 1, 2)
plt.imshow(unwarp_img)
'''

fig10 = plt.figure(figsize=(8, 4))

plt.subplot(2,1,1)
plt.imshow(res_img)
plt.subplot(2,1,2)

plt.imshow(unwarp_img)
plt.show()




# cv2.imwrite('warped_imgs/warp000' + pic_num + '.png', warped_img)
# cv2.imwrite('roi_imgs/roi2000' + pic_num + '.png', crop_img)
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
