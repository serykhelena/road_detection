from __future__ import print_function
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import math


def nothing(x):
    pass

warped = cv2.imread('output_images/warped1_1.png')
hsv_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

black_win_name = 'black'
green_win_name = 'green'
green_hsv_win_name = 'green hsv'

windows_mode = cv2.WINDOW_NORMAL
# blackeate window 'Settings'
cv2.namedWindow(black_win_name, windows_mode)
# create 6 trackbars for hsv-settings (black lane)
cv2.createTrackbar('h0', black_win_name, 0, 180, nothing)
cv2.createTrackbar('s0', black_win_name, 0, 255, nothing)
cv2.createTrackbar('v0', black_win_name, 0, 255, nothing)
cv2.createTrackbar('h1', black_win_name, 180, 180, nothing)
cv2.createTrackbar('s1', black_win_name, 255, 255, nothing)
cv2.createTrackbar('v1', black_win_name, 255, 255, nothing)

cv2.namedWindow(green_win_name, windows_mode)
# create 6 trackbars for hsv-settings (black lane)
cv2.createTrackbar('h0g', green_win_name, 0, 180, nothing)
cv2.createTrackbar('s0g', green_win_name, 0, 255, nothing)
cv2.createTrackbar('v0g', green_win_name, 0, 255, nothing)
cv2.createTrackbar('h1g', green_win_name, 180, 180, nothing)
cv2.createTrackbar('s1g', green_win_name, 255, 255, nothing)
cv2.createTrackbar('v1g', green_win_name, 255, 255, nothing)

while (1):
    # get value of hsv-settings (black lane)
    h0 = cv2.getTrackbarPos('h0', black_win_name)
    s0 = cv2.getTrackbarPos('s0', black_win_name)
    v0 = cv2.getTrackbarPos('v0', black_win_name)
    h1 = cv2.getTrackbarPos('h1', black_win_name)
    s1 = cv2.getTrackbarPos('s1', black_win_name)
    v1 = cv2.getTrackbarPos('v1', black_win_name)

    # get value of hsv-settings (green lane)
    h0g = cv2.getTrackbarPos('h0g', green_win_name)
    s0g = cv2.getTrackbarPos('s0g', green_win_name)
    v0g = cv2.getTrackbarPos('v0g', green_win_name)
    h1g = cv2.getTrackbarPos('h1g', green_win_name)
    s1g = cv2.getTrackbarPos('s1g', green_win_name)
    v1g = cv2.getTrackbarPos('v1g', green_win_name)

    lower_black = np.array([h0, s0, v0])
    upper_black = np.array([h1, s1, v1])

    lower_green = np.array([h0g, s0g, v0g])
    upper_green = np.array([h1g, s1g, v1g])

    mask_black = cv2.inRange(hsv_warped, lower_black, upper_black)
    mask_green = cv2.inRange(hsv_warped, lower_green, upper_green)
    frame = cv2.bitwise_or(mask_black, mask_green)
    cv2.imshow('green', mask_green)
    cv2.imshow('black', mask_black)
    cv2.imshow('test', frame)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()