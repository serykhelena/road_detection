#!/usr/bin/python

import cv2
import numpy as np
import unicorn
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg



inp_img = cv2.imread('input_imgs/frame000681.png')
inp_mask = cv2.imread('input_masks/frame000681_mask.pgm')
inp_mask_new_size = cv2.resize(inp_mask, (640, 480), interpolation=cv2.INTER_CUBIC)

height = inp_mask_new_size.shape[0]
width  = inp_mask_new_size.shape[1]

print("This image is: ", type(inp_mask_new_size),
      "Height is: ", height,
      "Width is: ", width)

crop_img = unicorn.find_low_border_of_roi(inp_mask_new_size, roi_window=10, x_limit=60, show_border=0)
crop_img_lined = unicorn.find_low_border_of_roi(inp_mask_new_size, roi_window=10, x_limit=60, show_border=1)

gray_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
canny_img = cv2.Canny(gray_img, 100, 200)
canny_rgb = cv2.cvtColor(canny_img, cv2.COLOR_GRAY2BGR)


ref_pts, ref_crop = unicorn.get_points(canny_rgb)


hough_lines = cv2.HoughLinesP(
    canny_img,
    rho=6,
    theta=np.pi / 60,
    threshold=50,
    lines=np.array([]),
    minLineLength=20,
    maxLineGap=10
)

color_lined_img = unicorn.draw_lines(canny_rgb, hough_lines)

### Sorting lines with thresholds   ###

left_line_x = []
left_line_y = []
limit_left_line = []
left_limit_x = 260

right_line_x = []
right_line_y = []
limit_right_line = []
right_limit_x = 395

center_line_x = []
center_line_y = []
center_line_all = []
center_left_limit_x = left_limit_x + 1
center_right_limit_x = right_limit_x - 1

for line in hough_lines:
    if line[0][0] <= left_limit_x and line[0][2] <= left_limit_x:
        left_line_x.append(line[0][0])
        left_line_x.append(line[0][2])

        left_line_y.append(line[0][1])
        left_line_y.append(line[0][3])

        limit_left_line.append(line)

    elif line[0][0] >= right_limit_x and line[0][2] >= right_limit_x:
        right_line_x.append(line[0][0])
        right_line_x.append(line[0][2])

        right_line_y.append(line[0][1])
        right_line_y.append(line[0][3])

        limit_right_line.append(line)

    elif line[0][0] < right_limit_x and line[0][2] < right_limit_x:
        if line[0][0] > left_limit_x and line[0][2] > left_limit_x:
            center_line_x.append(line[0][0])
            center_line_x.append(line[0][2])

            center_line_y.append(line[0][1])
            center_line_y.append(line[0][3])

            center_list = np.array([line[0][0], line[0][1], line[0][2], line[0][3]])
            center_line_all.append([center_list])

center_line = np.array(center_line_all)

left_limit_draw = unicorn.draw_lines(
    canny_rgb,
    limit_left_line,
    color=[255, 255, 0],
    thickness=5
)

right_limit_draw = unicorn.draw_lines(
    left_limit_draw,
    limit_right_line,
    color=[255, 0, 255],
    thickness=5
)

center_limit_draw = unicorn.draw_lines(
    right_limit_draw,
    center_line_all,
    color=[0, 255, 255],
    thickness=5
)
########################################################################################3

### Sorting lines with slope   ###

k_left_x = []
k_left_y = []
k_left = []
k_left_tg = []
k_left_rad = []
k_left_deg = []

k_right_x = []
k_right_y = []
k_right = []
k_right_tg = []
k_right_rad = []
k_right_deg = []

k_center_x = []
k_center_y = []
k_center = []
k_center_tg = []
k_center_rad = []
k_center_deg = []

k_tg = []
k_rad = []
k_deg = []

k_tg, k_rad, k_deg = unicorn.k_get(hough_lines)

for k in range(0, len(k_tg)):
    if k_tg[k] < 0:
        k_left_tg.append(k_tg[k])
        k_left_x.append(hough_lines[k][0][0])
        k_left_x.append(hough_lines[k][0][2])
        k_left_y.append(hough_lines[k][0][1])
        k_left_y.append(hough_lines[k][0][3])
        k_left.append(hough_lines[k])
    elif k_tg[k] > 0 and k_tg[k] < 1.5:
        k_right_tg.append(k_tg[k])
        k_right_x.append(hough_lines[k][0][0])
        k_right_x.append(hough_lines[k][0][2])
        k_right_y.append(hough_lines[k][0][1])
        k_right_y.append(hough_lines[k][0][3])
        k_right.append(hough_lines[k])
    elif k_tg[k] >= 1.5:
        k_center_tg.append(k_tg[k])
        k_center_x.append(hough_lines[k][0][0])
        k_center_x.append(hough_lines[k][0][2])
        k_center_y.append(hough_lines[k][0][1])
        k_center_y.append(hough_lines[k][0][3])
        k_center.append(hough_lines[k])

k_left_draw = unicorn.draw_lines(
    canny_rgb,
    k_left,
    color=[0, 255, 0],
    thickness=5
)

k_right_draw = unicorn.draw_lines(
    k_left_draw,
    k_right,
    color=[255, 0, 0],
    thickness=5
)

k_center_draw = unicorn.draw_lines(
    k_right_draw,
    k_center,
    color=[0, 0, 255],
    thickness=5
)
####################################################################

min_y = 0
max_y = crop_img.shape[0]

min_y_left = min(left_line_y)
max_y_left = max(left_line_y)

poly_left = np.poly1d(np.polyfit(
    left_line_y,
    left_line_x,
    deg=1
))
left_x_start = int(poly_left(max_y_left))
left_x_end = int(poly_left(min_y_left))

min_y_right = min(right_line_y)
max_y_right = max(right_line_y)

poly_right = np.poly1d(np.polyfit(
    right_line_y,
    right_line_x,
    deg=1
))
right_x_start = int(poly_right(max_y_right))
right_x_end = int(poly_right(min_y_right))

min_center_x = min(center_line_x)
max_center_y = max(center_line_y)

min_center_y = min(center_line_y)
max_center_x = max(center_line_x)

three_line_img = unicorn.draw_lines(
    canny_rgb,
    [[
        [left_x_start, max_y_left, left_x_end, min_y_left],
        [max_center_x, max_center_y, min_center_x, min_center_y],
        [right_x_start, max_y_right, right_x_end, min_y_right]
    ]],
    color=[255, 0, 0],
    thickness=5
)

apprx_left_line = [left_x_start, max_y_left, left_x_end, min_y_left]
apprx_center_line = [max_center_x, max_center_y, min_center_x, min_center_y]
apprx_right_line = [right_x_start, max_y_right, right_x_end, min_y_right]


combine_line = np.bitwise_or(three_line_img, ref_crop)
print(ref_pts)
ref_right_line = [ref_pts[0][1][0], ref_pts[0][1][1], ref_pts[0][0][0], ref_pts[0][0][1]]
ref_center_line = [ref_pts[1][1][0], ref_pts[1][1][1], ref_pts[1][0][0], ref_pts[1][0][1]]
ref_left_line = [ref_pts[2][1][0], ref_pts[2][1][1], ref_pts[2][0][0], ref_pts[2][0][1]]

left_k, left_b = unicorn.get_k_b_line(apprx_left_line)
right_k, right_b = unicorn.get_k_b_line(apprx_right_line)
cntr_k, cntr_b = unicorn.get_k_b_line(apprx_center_line)

ref_left_k, ref_left_b = unicorn.get_k_b_line(ref_left_line)
ref_right_k, ref_right_b = unicorn.get_k_b_line(ref_right_line)
ref_cntr_k, ref_cntr_b = unicorn.get_k_b_line(ref_center_line)


print("Left coo:", apprx_left_line, "k:", left_k, "b:", left_b)
print("Center coo:", apprx_center_line,"k:", cntr_k, "b:", cntr_b)
print("Right coo:", apprx_right_line, "k:", right_k, "b:", right_b)

print("Ref left:", ref_left_line, "k:", ref_left_k, "b:", ref_left_b)
print("Ref center:", ref_center_line, "k:", ref_cntr_k, "b:", ref_cntr_b)
print("Ref right:", ref_right_line, "k:", ref_right_k, "b:", ref_right_b)

dif_left_k = round(abs(ref_left_k) - abs(left_k), 2)
dif_center_k = round(abs(ref_cntr_k) - abs(cntr_k), 2)
dif_right_k = round(abs(ref_right_k) - abs(right_k), 2)


dif_left_b = round(abs(ref_left_b - left_b), 2)
dif_center_b = round(abs(ref_cntr_b - cntr_b), 2)
dif_right_b = round(abs(ref_right_b - right_b), 2)

font = cv2.FONT_HERSHEY_SIMPLEX

cv2.putText(combine_line, str(dif_left_k), (20, 50), font, 1, (255, 0, 0), thickness=2)
cv2.putText(combine_line, str(dif_center_k), (320, 50), font, 1, (255, 0, 0), thickness=2)
cv2.putText(combine_line, str(dif_right_k), (520, 50), font, 1, (255, 0, 0), thickness=2)

print("D_k left", dif_left_k, "D_b left", dif_left_b)
print("D_k center", dif_center_k, "D_b center", dif_center_b)
print("D_k right", dif_right_k, "D_b right", dif_right_b)

fig = plt.figure(figsize=(8, 8))

# plt.imshow(ref_crop)
fig.add_subplot(3, 1, 1)
plt.imshow(ref_crop)
fig.add_subplot(3, 1, 2)
plt.imshow(three_line_img)
fig.add_subplot(3, 1, 3)
plt.imshow(combine_line)
# fig.add_subplot(5, 1, 4)
# plt.imshow(center_limit_draw)
# fig.add_subplot(5, 1, 5)
# plt.imshow(crop_img)

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('roi_imgs/roi000681.png', crop_img)

cv2.imwrite('roi_imgs/roi000681_lined.png', crop_img_lined)
cv2.imwrite('first_method_output/compare000681.png', combine_line)


# cv2.imwrite('sort_lines/slope000000.png', k_center_draw)
