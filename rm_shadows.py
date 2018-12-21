#!/usr/bin/python
from __future__ import print_function
import cv2
import numpy as np
from shapely.geometry import LineString
from shapely.geometry import Point
# import math
from math import atan2, tan, degrees, sqrt, acos, cos, atan

btn_down = False

def get_points(img):
    # Create dictionary for my_mouse func
    data = {}
    data['img'] = img.copy()
    data['lines'] = []
    data['right_line'] = []
    data['left_line'] = []
    data['center_line'] = []
    # Set the callback function for any mouse event
    cv2.line(img, (0, int(img.shape[0] * 0.15)), (img.shape[1], int(img.shape[0] * 0.15)), (127, 0, 255), thickness=2)

    cv2.imshow('test', img)
    cv2.setMouseCallback('test', my_mouse_left_click, data)

    cv2.waitKey(0)
    # Convert array to np.array in shape n,2,2
    points = np.uint16(data['lines'])
    left_ref_line = []
    center_ref_line = []
    right_ref_line = []
    for line in range(0, 3):
        # left reference line
        if points[line][0][0] < 240:
            for k in range(0, 2):
                for j in range(0, 2):
                    left_ref_line.append(points[line][k][j])
        # center reference line
        if points[line][0][0] >= 240 and points[line][0][0] <= 400:
            for k in range(0, 2):
                for j in range(0, 2):
                    center_ref_line.append(points[line][k][j])
        # right reference line
        if points[line][0][0] >= 460:
            for k in range(0, 2):
                for j in range(0, 2):
                    right_ref_line.append(points[line][k][j])

    left_ref_line = np.uint16(left_ref_line)
    center_ref_line = np.uint16(center_ref_line)
    right_ref_line = np.uint16(right_ref_line)

    return points, data['img'], left_ref_line, center_ref_line, right_ref_line


def my_mouse_left_click(event, x, y, flags, data):
    global btn_down
    cv2.line(data['img'], (0, int(data['img'].shape[0] * 0.15)), (data['img'].shape[1], int(data['img'].shape[0] * 0.15)), (127, 0, 255), thickness=2)
    if event == cv2.EVENT_LBUTTONDOWN and btn_down:
        btn_down = False
        # add second point
        data['lines'][0].append((x, y))

        cv2.circle(data['img'], (x, y), 5, (255, 0, 255), thickness=-1)
        cv2.line(data['img'], data['lines'][0][0], data['lines'][0][1], (255, 0, 255), thickness=2)

        cv2.imshow('test', data['img'])
        # print("lines_last", data['lines'])


    elif event == cv2.EVENT_MOUSEMOVE and btn_down:
        # this is just for a ine visualization
        new_image = data['img'].copy()
        cv2.line(new_image, data['lines'][0][0], (x, y), (255, 0, 255), 2)
        cv2.imshow("test", new_image)

    elif event == cv2.EVENT_LBUTTONDOWN and len(data['lines']) < 3:
        btn_down = True
        data['lines'].insert(0, [(x, y)])  # prepend the point
        cv2.circle(data['img'], (x, y), 5, (255, 0, 255), thickness=-1) # first point
        cv2.imshow("test", data['img'])

def region_of_interest(img, vertices):
    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(img)   # black image with origin image sizes
    # Return the number of color channels of the image.
    #channel_count = img.shape[2]
    # Array with white colour (255, 255, 255)
    #match_mask_color = (255,) * channel_count
    match_mask_color = 255  # <-- This line altered for grayscale.
    # Fill inside the polygon
    # all black except roi (via vertices)
    cv2.fillPoly(mask, vertices, match_mask_color)
    # Returning the image only where mask pixels match
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

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

def get_angle_and_const(line):
    x_diff = line[0] - line[2]
    y_diff = line[1] - line[3]
    k = y_diff / float(x_diff)
    k_deg = degrees(atan2(y_diff, x_diff))
    b = line[1] - k * line[0]
    return k, k_deg, b

def get_left_angle_and_const(line):
    x_diff = line[2] - line[0]
    y_diff = line[1] - line[3]
    k = y_diff / float(x_diff)
    k_deg = degrees(atan2(y_diff, x_diff))
    b = line[1] - k * line[0]
    return k, k_deg, b

# Load BGR image
origin = cv2.imread('pics_frame1_x/frame1_9.jpg', 1)
# Load image in grayscale
image = cv2.imread("pics_frame1_x/frame1_9.jpg", 0)

min_y = int(origin.shape[0] * 0.15)

# Dilate the image to get rid of lines
dilate_im = cv2.dilate(image, np.ones((3, 3), np.uint8))
# Median blur - to get image with only shadows
blur_im = cv2.medianBlur(dilate_im, 31)
# Calculate the difference between original and obtained image. Reverse colours
diff_im = 255 - cv2.absdiff(image, blur_im)

canny_img   = cv2.Canny(diff_im, 100, 400)

roi_vertices = [
    (0, diff_im.shape[0]),
    (0, int(diff_im.shape[0]/2.5)),
    (0+150, int(diff_im.shape[0]/8)),
    (diff_im.shape[1]-130, int(diff_im.shape[0]/8)),
    (diff_im.shape[1], int(diff_im.shape[0]/2.5)),
    (diff_im.shape[1], diff_im.shape[0])
]
# cv2.imshow('image', image)
# cv2.imshow('dilate', dilate_im)
# cv2.imshow('blur', blur_im)
# cv2.imshow('diff', diff_im)
# cv2.imshow('canny', canny_img)
# draw roi points
for i in range(0, len(roi_vertices)):
    cv2.circle(diff_im, roi_vertices[i], 5, (0, 255, 0), thickness=-1)

# Cropping to Region of Interest (ROI)
roi_image = region_of_interest(
    canny_img,
    np.array([roi_vertices], np.int32) # from list make numpy array
)

cv2.imshow('ROI', roi_image)

# array with lines [[x1, y1, x2, y2]]
lines = cv2.HoughLinesP(
    roi_image,
    rho=6,
    theta=np.pi/60,
    threshold=80,
    lines=np.array([]),
    minLineLength=15,
    maxLineGap=8
)

line_image = draw_lines(
    origin,
    lines,
    thickness=5
)

cv2.imshow('lined', line_image)
# cv2.waitKey(0)
left_line = []
left_line_x = []
left_line_y = []

right_line = []
right_line_x = []
right_line_y = []

center_line = []
center_line_x = []
center_line_y = []

for line in lines:
    for x1, y1, x2, y2 in line:
        if x1 < 240 and x2 < 240:
            left_line.extend([x1, y1, x2, y2])
            left_line_x.extend([x1, x2])
            left_line_y.extend([y1, y2])
        elif x1 > 280 and x1 < 400 and x2 > 280 and x2 < 400:
            center_line.extend([x1, y1, x2, y2])
            center_line_x.extend([x1, x2])
            center_line_y.extend([y1, y2])
        elif x1 > 460 and x2 > 460:
            right_line.extend([x1, y1, x2, y2])
            right_line_x.extend([x1, x2])
            right_line_y.extend([y1, y2])
'''
for i in range(0, len(left_line)-3, 4):
    cv2.line(origin, (left_line[i], left_line[i+1]), (left_line[i+2], left_line[i+3]), (255, 0, 0), thickness=3)

for i in range(0, len(center_line)-3, 4):
    cv2.line(origin, (center_line[i], center_line[i+1]), (center_line[i+2], center_line[i+3]), (0, 255, 0), thickness=3)

for i in range(0, len(right_line)-3, 4):
    cv2.line(origin, (right_line[i], right_line[i+1]), (right_line[i+2], right_line[i+3]), (0, 0, 255), thickness=3)
'''
min_y = int(origin.shape[0] * 0.15)
max_y = origin.shape[0]

poly_center = np.poly1d(np.polyfit(
    center_line_y,
    center_line_x,
    deg=1
))

center_x_start = int(poly_center(max_y))
center_x_end = int(poly_center(min_y))

poly_left = np.poly1d(np.polyfit(
    left_line_y,
    left_line_x,
    deg=1
))

left_x_start = int(poly_left(max_y))
left_x_end = int(poly_left(min_y))

poly_left = np.poly1d(np.polyfit(
    right_line_y,
    right_line_x,
    deg=1
))

right_x_start = int(poly_left(max_y))
right_x_end = int(poly_left(min_y))


cv2.namedWindow('test', cv2.WINDOW_AUTOSIZE)
pts, final_img, left_r_line, center_r_line, right_r_line = get_points(origin)
right_r_k, right_r_k_deg, right_r_b = get_angle_and_const(right_r_line)
print('RIGHT R K    ', right_r_k, '     Right R K deg', right_r_k_deg, '    R R B', right_r_b)

left_r_k, left_r_k_deg, left_r_b = get_left_angle_and_const(left_r_line)
print('LEFT R K     ', left_r_k, '      LEFT R K deg', left_r_k_deg, '    L R B', left_r_b)

center_r_k, center_r_k_deg, center_r_b = get_angle_and_const(center_r_line)
print('CENTER R K   ', center_r_k, '     CENTER R K deg', center_r_k_deg, '   C R B', center_r_b)

cv2.line(final_img, (center_x_start, max_y), (center_x_end, min_y), (0, 255, 0), thickness=3)
cv2.line(final_img, (left_x_start, max_y), (left_x_end, min_y), (255, 0, 0), thickness=3)
cv2.line(final_img, (right_x_start, max_y), (right_x_end, min_y), (0, 255, 255), thickness=3)
left_fin_line   = np.int16([left_x_start, max_y, left_x_end, min_y])
left_fin_k, left_fin_k_deg, left_fin_b = get_left_angle_and_const(left_fin_line)
print('LEFT FIN K   ', left_fin_k, '    LEFT FIN K deg', degrees(atan(left_fin_k)), '   L F B', left_fin_b)

right_fin_line  = np.int16([right_x_start, max_y, right_x_end, min_y])
right_fin_k, right_fin_k_deg, right_fin_b = get_angle_and_const(right_fin_line)
print('RIGHT FIN K  ', right_fin_k, '   RIGHT FIN K deg', degrees(atan(right_fin_k)), ' R F B', right_fin_b)

center_fin_line = np.int16([center_x_start, max_y, center_x_end, min_y])
center_fin_k, center_fin_k_deg, center_fin_b = get_angle_and_const(center_fin_line)
print('CENTER FIN K ', center_fin_k, '  CENTER FIN K deg', degrees(atan(center_fin_k)), '   C F B', center_fin_b)

# left line
cv2.putText(final_img, ("b=" + str(round(abs(left_fin_b - left_r_b), 2))), (50, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
cv2.putText(final_img, 'k=' + str(round(abs(left_fin_k_deg - left_r_k_deg), 2)), (50, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

# right line
cv2.putText(final_img, "b=" + str(round(abs(right_fin_b - right_r_b), 2)), (420, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
cv2.putText(final_img, 'k=' + str(round(abs(right_fin_k_deg - right_r_k_deg), 2)), (420, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)


#center line
cv2.putText(final_img, "b=" + str(round(abs(center_fin_b - center_r_b), 2)), (250, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
cv2.putText(final_img, 'k=' + str(round(abs(center_fin_k_deg - center_r_k_deg), 2)), (250, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)


cv2.imshow('test', final_img)
cv2.waitKey(0)

cv2.imwrite('new_frame1_x/new_frame1_9.png', final_img)

cv2.destroyAllWindows()
print('Finished successfully!')

