from __future__ import print_function
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import math
import pickle


def undistort_image(img, objpts, imgpts):
    """
    Returns an undistorted image
    The desired object and image points must also be supplied to this function
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, gray.shape[::-1], None, None)
    # print('un', img.shape)
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (img.shape[1], img.shape[0]), 1, (img.shape[1], img.shape[0]))
    # print('newcamera', newcameramtx)
    # cv2.imshow('roi', roi)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    # If there are no lines to draw, exit.
    if lines is None:
        return
    # Make a copy of the original image.
    img = np.copy(img)
    # Create a blank image that matches the original in size.
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8,
    )
    # Loop over all lines and draw them on the blank image.
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    # Merge the image with the lines onto the original.
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    # Return the modified image.
    return img


def compute_hls_black_green_binary(rgb_img):
    """
    Returns a binary thresholded image produced retaining only black and green elements on the picture
    The provided image should be in RGB format
    """
    hls_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HLS)
    h0 = 0
    h1 = 23
    l0 = 6
    l1 = 50
    s0 = 34
    s1 = 101
    lower_black = np.array([h0, l0, s0])
    upper_black = np.array([h1, l1, s1])
    h0g = 52
    h1g = 80
    l0g = 37
    l1g = 151
    s0g = 0
    s1g = 75
    lower_green = np.array([h0g, l0g, s0g])
    upper_green = np.array([h1g, l1g, s1g])

    mask_black = cv2.inRange(hls_img, lower_black, upper_black)
    mask_green = cv2.inRange(hls_img, lower_green, upper_green)

    img_hls_black_green_bin = cv2.bitwise_or(mask_black, mask_green)

    return img_hls_black_green_bin


def abs_sobel(gray_img, x_dir=True, kernel_size=3, thres=(0, 255)):
    """
    Applies the sobel operator to a grayscale-like (i.e. single channel) image in either horizontal or vertical direction
    The function also computes the asbolute value of the resulting matrix and applies a binary threshold
    """
    sobel = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=kernel_size) if x_dir else cv2.Sobel(gray_img, cv2.CV_64F, 0, 1,
                                                                                             ksize=kernel_size)
    sobel_abs = np.absolute(sobel)
    sobel_scaled = np.uint8(255 * sobel / np.max(sobel_abs))

    gradient_mask = np.zeros_like(sobel_scaled)
    gradient_mask[(thres[0] <= sobel_scaled) & (sobel_scaled <= thres[1])] = 1
    return gradient_mask

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def nothing(x):
	pass

def select_roi(undistort_image):
    undist_img_width = undistort_image.shape[1]
    undist_img_height = undistort_image.shape[0]
    half_img_width = int(undist_img_width / 2)

    win_name = 'roi_selection'
    windows_mode = cv2.WINDOW_NORMAL
    cv2.namedWindow(win_name, windows_mode)

    cv2.createTrackbar('up_left', win_name, 0, half_img_width-20, nothing)
    cv2.createTrackbar('up_right', win_name, 0, half_img_width+40, nothing)

    cv2.createTrackbar('down', win_name, 0, undist_img_height, nothing)
    cv2.createTrackbar('down_left', win_name, 0, half_img_width, nothing)
    cv2.createTrackbar('down_right', win_name, 0, half_img_width, nothing)
    while True:
        temp_img = undistort_image.copy()
        up_left = cv2.getTrackbarPos('up_left', win_name)
        up_right = cv2.getTrackbarPos('up_right', win_name)
        down = cv2.getTrackbarPos('down', win_name)
        down_left = cv2.getTrackbarPos('down_left', win_name)
        down_right = cv2.getTrackbarPos('down_right', win_name)

        cv2.line(temp_img, (half_img_width - down_left, down), (half_img_width + down_right, down), (0, 0, 255), thickness=3)
        cv2.line(temp_img, (half_img_width-20 - up_left, 0), (up_right + half_img_width-20, 0), (0, 0, 255), thickness=3)

        cv2.line(temp_img, (half_img_width - down_left, down), (half_img_width-20 - up_left, 0), (0, 0, 255), thickness=3)
        cv2.line(temp_img, (half_img_width + down_right, down), (up_right + half_img_width-20, 0), (0, 0, 255), thickness=3)
        cv2.imshow(win_name, temp_img)
        pnt1 = [half_img_width-20 - up_left, 0]        # upper left
        pnt2 = [up_right + half_img_width-20, 0]       # upper right
        pnt3 = [half_img_width + down_right, down]  # lower right
        pnt4 = [half_img_width - down_left, down]   # lower left
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            cut_img = undistort_image[pnt1[1]:pnt4[1], pnt4[0]:pnt3[0]]
            return pnt1, pnt2, pnt3, pnt4, cut_img
            break

def cut_img(undistorted_image):
    undist_img_width = undistorted_image.shape[1]
    undist_img_height = undistorted_image.shape[0]
    win_name = 'cut_selection'
    windows_mode = cv2.WINDOW_NORMAL
    cv2.namedWindow(win_name, windows_mode)

    cv2.createTrackbar('height', win_name, undist_img_height, undist_img_height, nothing)
    cv2.createTrackbar('width', win_name, 0, undist_img_width, nothing)

    while True:
        height = cv2.getTrackbarPos('height', win_name)
        width = cv2.getTrackbarPos('width', win_name)

        cut_img = undistorted_image[0:height, width: undist_img_width]
        cv2.imshow(win_name, cut_img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            return cut_img
            break

def fix_illumination(img, clip_val = 2, grid_size = (8, 8)):
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_img)
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=clip_val, tileGridSize=grid_size)
    cl = clahe.apply(l_channel)
    merge_channels = cv2.merge((cl, a_channel, b_channel))
    res_img = cv2.cvtColor(merge_channels, cv2.COLOR_LAB2BGR)
    return res_img

def k_get(lines):
    k_tg = []
    k_rad = []
    k_deg = []
    for line in range(0, len(lines)):
        new_line = lines[line][0]
        dx = new_line[2] - new_line[0]
        dy = new_line[3] - new_line[1]
        if dx == 0:
            if dy > 0:
                k_temp = 90  # 90 !!!! LIE, k = tg(), not angle in rad!!!!!!
                k_tg.append(k_temp)
                k_rad.append(1.5708)
                k_deg.append(90)
            if dy < 0:
                k_temp = -90
                k_tg.append(k_temp)
                k_rad.append(-1.5708)
                k_deg.append(-90)
        else:
            k_temp = round(dy / float(dx), 4)
            k_tg.append(k_temp)
            k_rad.append(math.atan2(dy, dx))
            k_deg.append(math.degrees(math.atan2(dy, dx)))

    return k_tg, k_rad, k_deg


with open('cal_param_opts.txt', 'rb') as fp:
    opts = pickle.load(fp)

with open('cal_param_ipts.txt', 'rb') as fp:
    ipts = pickle.load(fp)

image = cv2.imread('pics_one_lane_x/frame1_1.png')
undist_img = undistort_image(image, opts, ipts)

pnt1, pnt2, pnt3, pnt4, cut_image = select_roi(undist_img)
src_pts = np.array([pnt1, pnt2, pnt3, pnt4], dtype = np.float32)
dst_pts = np.array([[0, 0], [undist_img.shape[1], 0], [undist_img.shape[1], undist_img.shape[0]], [0, undist_img.shape[0]]], dtype = np.float32)
m = cv2.getPerspectiveTransform(src_pts, dst_pts)
warped = cv2.warpPerspective(undist_img, m, (undist_img.shape[1], undist_img.shape[0]))   # (maxWidth, maxHeight))
cv2.imshow('warepd', warped)

# warped = cv2.imread('output_images/warped1_1.png')
res_img = fix_illumination(warped)
canny_img = cv2.Canny(res_img, 10, 100)

lines = cv2.HoughLinesP(
    canny_img,
    rho=6,
    theta=np.pi/10,
    threshold=90,
    lines=np.array([]),
    minLineLength=25,
    maxLineGap=25
)

line_image = draw_lines(
    res_img,
    lines,
    thickness=5
)

k_tg, k_rad, k_deg = k_get(lines)
print('k_tg', k_tg)
print('k_deg', k_deg)
#
good_lines = []
good_k_tg = []
for koef in range(0, len(k_deg)):
    if (k_deg[koef] >= 45 and k_deg[koef] <= 135) or (k_deg[koef] <= -45 and k_deg[koef] >= -135):
        good_lines.append(lines[koef][0])
        good_k_tg.append(k_tg[koef])

line_image = draw_lines(
    res_img,
    lines,
    thickness=5
)

cv2.imshow('canny', canny_img)
cv2.imshow('lined', line_image)

k = cv2.waitKey(0) & 0xFF
if k == 27: # k == ESC
    cv2.destroyAllWindows()
elif k == ord('s'): # k == s -> save and exit
    cv2.imwrite('output_images/warped1_1.png', warped)
    cv2.imwrite('output_images/canny1_1.png', canny_img)
    cv2.imwrite('output_images/Hough1_1.png', line_image)
    cv2.destroyAllWindows()



'''



useful_lines = []
k_lines = []
for line in range(0, len(lines)):
    new_line = lines[line][0]
    # dx = abs(new_line[2] - new_line[0])
    # dy = abs(new_line[3] - new_line[1])
    dx = new_line[2] - new_line[0]
    dy = new_line[3] - new_line[1]
    if dx == 0:
        k = 1.5708 # 90 !!!! LIE, k = tg(), not angle in rad!!!!!!
        k_deg = 90
    else:
        k = round(dy / float(dx), 4)
        k_deg = math.degrees(math.atan2(dy, dx))

    k_lines.append(k)
    if k == 1.5708:
    # if k >= 1.4835 and k <= 1.6581: # 85 < k < 95
        useful_lines.append(new_line)
useful_lines = np.array([useful_lines])
# print('useful count', len(useful_lines))

repeated_k_lines = []

useful_img = draw_lines(
    warped,
    useful_lines,
    thickness=5
)

cv2.imshow('useful', useful_img)


unrepeated_k_lines = np.unique(k_lines)
paral_k = []
for u in range(0, len(unrepeated_k_lines)):
    counter_rep = 0
    for l in range(0, len(k_lines)):
        if k_lines[l] == unrepeated_k_lines[u]:

            counter_rep += 1
            if counter_rep == 2:
                paral_k.append(temp)
                paral_k.append(k_lines[l])
            if counter_rep > 2:
                paral_k.append(k_lines[l])
            temp = k_lines[l]

unique_paral_k = np.unique(paral_k)

paral_lines = []
for sk in paral_k:
    for kkk in range(0, len(k_lines)):
        if k_lines[kkk] == sk:
            paral_lines.append(lines[kkk])

paral_lines_img = draw_lines(
    warped,
    paral_lines,
    thickness=5
)
cv2.imshow('paral_img', paral_lines_img)
print('uniq', unique_paral_k)
print('par_k', paral_k)

paral_b = []
for lsl in range(0, len(paral_k)):
    temp_b = paral_lines[lsl][0][1] - paral_k[lsl]*paral_lines[lsl][0][0] # b = y - k*x
    paral_b.append(round(temp_b, 4))

amount_arr_k = len(unique_paral_k)
sorted_lines = np.zeros((amount_arr_k, 1), np.float64)
# sorted_k = np.zeros((amount_arr_k, 1), np.float64)
sorted_b = np.zeros((amount_arr_k, 1), np.float64)

# print('TEST', np.insert(sorted_lines[0], len(sorted_lines[0]), 5))
# print(sorted_lines)
# np.insert(sorted_k[0], 0, paral_k[0])

col_number = []
# count_col = 0
for amm in range(0, amount_arr_k):
    count_col = 0
    for kk in range(0, len(paral_k)):
        if paral_k[kk] == unique_paral_k[amm]:
            count_col += 1
    col_number.append(count_col)
max_col = max(col_number)
# print('max', max(col_number))
print('c_n', col_number)

# print('pl', type(paral_lines[0][0]), paral_lines[1][0])
sorted_k = np.ndarray(amount_arr_k, dtype=object)
print('init', sorted_k)
# sorted_k = np.zeros((amount_arr_k, int(max_col)), np.float64)
# sorted_lines = np.zeros((amount_arr_k, int(max_col)), np.ndarray)
for am in range(0, amount_arr_k):
    count_sort = 0
    # temp_k = np.zeros((1, col_number[am]))
    print(am)
    temp_k = []
    for k in range(0, len(paral_k)):
        if paral_k[k] == unique_paral_k[am]:
            # temp_k[0][count_sort] = paral_k[k]
            temp_k.append(paral_k[k])
            # sorted_lines[am][count_sort] = [paral_lines[k][0]]
            count_sort += 1
    print(temp_k)
    sorted_k[am] = temp_k
    print('sk', sorted_k)
    # if len(sorted_k) < max_col:
    #     sorted_k[am][count_sort:] = -10
        # sorted_lines[am][count_sort:] = -10
    # temp_k = temp_k
    # print('temp', temp_k)
    #
    # if am == 0:
    #     sorted_k = temp_k
    #     print('sorted 0', sorted_k)
    # else:
    #     # sorted_k = np.vstack((sorted_k, temp_k))
    #     # sorted_k = np.concatenate((sorted_k, temp_k), axis = 0)
    #     sorted_k = np.concatenate((sorted_k, temp_k))
    #     print('sorted n', sorted_k)
print(sorted_k)
print('T', sorted_k.T)
# print(sorted_lines)

A = [1, 1, 3, 3]
B = [2, 1]
# B = np.concatenate((A, B), axis=0)
# B = np.stack(A, B)
print('check', B) #np.concatenate((A, B), axis=0))
# print('sorted_k', sorted_k)
# print('temp', temp)
# print('new', )


'''
'''
new_let_see_lines = []
new_lines_b = []

# for aa in range(0, len(test_k_0)):
aa = 0
new_let_see_lines.append(test_lines_0[aa])
new_lines_b.append(paral_b[aa])
for bb in range(aa+1, len(test_k_0)):
    ds = abs(paral_b[aa] - paral_b[bb]) * math.sin(3.14 / 2 - math.atan(test_k_0[aa]))
    if ds >= 10:
        new_let_see_lines.append(test_lines_0[bb])
        new_lines_b.append(paral_b[bb])
    if ds < 10:
        y1_temp = int(abs(paral_b[aa] + paral_b[bb]) / 2)
        x1_temp = 0
        y2_temp = int(test_k_0[bb] * paral_lines_img.shape[1] + int(abs(paral_b[aa] + paral_b[bb]) / 2))
        x2_temp = paral_lines_img.shape[1]
        # aver_lines.append([x1_temp, y1_temp, x2_temp, y2_temp])

        new_let_see_lines.append(np.array([[x1_temp, y1_temp, x2_temp, y2_temp]]).astype(np.int32))
        b_av = new_let_see_lines[bb][0][1] - test_k_0[bb] * new_let_see_lines[bb][0][0]
        new_lines_b.append(b_av)
        # print('len(', len(new_let_see_lines))
        if len(new_let_see_lines) >=2:
            aver_ds = abs(new_lines_b[bb-1] - new_lines_b[bb]) * math.sin(3.14 / 2 - math.atan(test_k_0[bb]))
            if aver_ds < 10:
                y1_av = int(abs(new_lines_b[bb-1] + new_lines_b[bb]) / 2)
                x1_av = 0
                y2_av = int(test_k_0[bb] * paral_lines_img.shape[1] + int(abs(new_lines_b[bb-1] + new_lines_b[bb]) / 2))
                x2_av = paral_lines_img.shape[1]
                new_let_see_lines[bb] = np.array([[x1_av, y1_av, x2_av, y2_av]]).astype(np.int32)

# cv2.line(paral_lines_img, (test_lines_0[0][0][0], test_lines_0[0][0][1]), (test_lines_0[0][0][2], test_lines_0[0][0][3]), (0, 0, 255), thickness=4)
# cv2.line(paral_lines_img, (test_lines_0[1][0][0], test_lines_0[1][0][1]), (test_lines_0[1][0][2], test_lines_0[1][0][3]), (0, 0, 255), thickness=4)
# cv2.line(paral_lines_img, (test_lines_0[2][0][0], test_lines_0[2][0][1]), (test_lines_0[2][0][2], test_lines_0[2][0][3]), (0, 255, 255), thickness=4)
# cv2.line(paral_lines_img, (test_lines_0[3][0][0], test_lines_0[3][0][1]), (test_lines_0[3][0][2], test_lines_0[3][0][3]), (255, 0, 255), thickness=4)

average_filter_img = draw_lines(
    warped,
    new_let_see_lines,
    thickness=5
)

# cv2.imshow('average filter', average_filter_img)
'''
'''
# cv2.imshow('new filter', paral_lines_img)


# filtered_line_image = draw_lines(
#     warped,
#     useful_lines,
#     thickness=5
# )
# cv2.imshow('strict filter', filtered_line_image)
'''
'''
dw = int(warped.shape[1] / 3)
left_lines = []
center_lines = []
right_lines = []

for u in range(0, len(useful_lines)):
    # print(useful_lines[u][0], useful_lines[u][2])
    if useful_lines[u][0] < dw and useful_lines[u][2] < dw:
        left_lines.append(useful_lines[u])
    elif useful_lines[u][0] < 2*dw and useful_lines[u][2] < 2*dw:
        if useful_lines[u][0] > dw and useful_lines[u][2] > dw:
            center_lines.append(useful_lines[u])
    else:
        right_lines.append(useful_lines[u])

# print('left line', left_lines)
# print('center line', center_lines)
# print('right line', right_lines)
left_x = []
for ll in range(0, len(left_lines)):
    left_x.append(left_lines[ll][0])
    left_x.append(left_lines[ll][2])
    cv2.line(filtered_line_image, (left_lines[ll][0], left_lines[ll][1]), (left_lines[ll][2], left_lines[ll][3]), (0, 255, 0), thickness=5)

min_left_x = min(left_x)
max_left_x = max(left_x)
left_dx = max_left_x + min_left_x
cv2.line(filtered_line_image, (int(left_dx/2), filtered_line_image.shape[0]), (int(left_dx/2), 0), (0, 0, 255), thickness=5)

center_x = []
for cc in range(0, len(center_lines)):
    center_x.append(center_lines[cc][0])
    center_x.append(center_lines[cc][2])
    cv2.line(filtered_line_image, (center_lines[cc][0], center_lines[cc][1]), (center_lines[cc][2], center_lines[cc][3]), (0, 0, 255), thickness=5)

min_center_x = min(center_x)
max_center_x = max(center_x)
center_dx = max_center_x + min_center_x

cv2.line(filtered_line_image, (int(center_dx/2), filtered_line_image.shape[0]), (int(center_dx/2), 0), (0, 255, 0), thickness=5)

right_x = []
for rr in range(0, len(right_lines)):
    right_x.append(right_lines[rr][0])
    right_x.append(right_lines[rr][2])
    cv2.line(filtered_line_image, (right_lines[rr][0], right_lines[rr][1]), (right_lines[rr][2], right_lines[rr][3]), (255, 0, 255), thickness=5)

min_right_x = min(right_x)
max_right_x = max(right_x)
right_dx = max_right_x + min_right_x

cv2.line(filtered_line_image, (int(right_dx/2), filtered_line_image.shape[0]), (int(right_dx/2), 0), (0, 255, 255), thickness=5)


'''

cv2.waitKey(0)


cv2.destroyAllWindows()