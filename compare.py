#!/usr/bin/python

import cv2
import numpy as np
import unicorn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg







inp_img     = cv2.imread('input_imgs/frame000000.png')
inp_mask    = cv2.imread('input_masks/frame000000_mask.pgm')
inp_mask_new_size = cv2.resize(inp_mask, (640, 480), interpolation=cv2.INTER_CUBIC)

height = inp_mask_new_size.shape[0]
width  = inp_mask_new_size.shape[1]

print("This image is: ", type(inp_mask_new_size),
      "Height is: ", height,
      "Width is: ", width)

crop_img = unicorn.find_low_border_of_roi(inp_mask_new_size, roi_window=10)

gray_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
canny_img = cv2.Canny(gray_img, 100, 200)
canny_rgb = cv2.cvtColor(canny_img, cv2.COLOR_GRAY2BGR)

hough_lines = cv2.HoughLinesP(
    canny_img,
    rho=6,
    theta=np.pi / 60,
    threshold=80,
    lines=np.array([]),
    minLineLength=5,
    maxLineGap=10
)

color_lined_img = unicorn.draw_lines(canny_rgb, hough_lines)

left_line_x = []
left_line_y = []
left_limit_x = 260

right_line_x = []
right_line_y = []
right_limit_x = 395


center_line_x = []
center_line_y = []
center_line_all = []
center_left_limit_x = left_limit_x + 1
center_right_limit_x = right_limit_x - 1


print("Left limit: ", left_limit_x,
      "Center limits: ", center_left_limit_x, center_right_limit_x,
      "Right limit: ", right_limit_x)

for line in hough_lines:
    if line[0][0] <= left_limit_x and line[0][2] <= left_limit_x:
        left_line_x.append(line[0][0])
        left_line_x.append(line[0][2])

        left_line_y.append(line[0][1])
        left_line_y.append(line[0][3])

    elif line[0][0] >= right_limit_x and line[0][2] >= right_limit_x:
        right_line_x.append(line[0][0])
        right_line_x.append(line[0][2])

        right_line_y.append(line[0][1])
        right_line_y.append(line[0][3])

    elif line[0][0] < right_limit_x and line[0][2] < right_limit_x:
        if line[0][0] > left_limit_x and line[0][2] > left_limit_x:
            center_line_x.append(line[0][0])
            center_line_x.append(line[0][2])

            center_line_y.append(line[0][1])
            center_line_y.append(line[0][3])

            center_list = np.array([line[0][0], line[0][1], line[0][2], line[0][3]])
            center_line_all.append([center_list])

center_line = np.array(center_line_all)

min_y = 0
max_y = crop_img.shape[0]

poly_left = np.poly1d(np.polyfit(
    left_line_y,
    left_line_x,
    deg=1
))
left_x_start = int(poly_left(max_y))
left_x_end = int(poly_left(min_y))

poly_right = np.poly1d(np.polyfit(
    right_line_y,
    right_line_x,
    deg=1
))
right_x_start = int(poly_right(max_y))
right_x_end = int(poly_right(min_y))

poly_center = np.poly1d(np.polyfit(
    center_line_x,
    center_line_y,
    deg=1
))
center_x_start = int(poly_center(max_y))
center_x_end = int(poly_center(min_y))

print("MH", center_x_start, center_x_end)


calc_center_line = [center_line_x[0], center_line_y[0], center_line_x[1], center_line_y[1]]
c_k, c_b = unicorn.get_k_b_line(calc_center_line)


sep_lined_img = unicorn.draw_lines(
    canny_rgb,
    [[
        [left_x_start, max_y, left_x_end, min_y],
        [right_x_start, max_y, right_x_end, min_y]
    ]],
    thickness=5
)

check_center = unicorn.draw_lines(
    canny_rgb,
    center_line_all
)

for line in center_line:
    print("Line: ", line)




min_center_x = min(center_line_x)
max_center_y = max(center_line_y)


min_center_y = min(center_line_y)
max_center_x = max(center_line_x)

average_center_x = int((min_center_x + max_center_x)/2)


print(min_center_x, max_center_y, max_center_x, min_center_y)
print(average_center_x, max_center_y, average_center_x, min_center_y)

cv2.line(sep_lined_img, (average_center_x, max_center_y), (average_center_x, min_center_y), color=(0, 255, 0), thickness=5)


fig = plt.figure(figsize=(8, 8))

fig.add_subplot(4, 1, 1)
plt.imshow(canny_img)
fig.add_subplot(4, 1, 2)
plt.imshow(color_lined_img)
fig.add_subplot(4, 1, 3)
plt.imshow(sep_lined_img)
fig.add_subplot(4, 1, 4)
plt.imshow(check_center)




plt.show()
# cv2.imwrite('hough_trfrm/hough000000.png', color_lined_img)
