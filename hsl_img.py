from __future__ import print_function
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import math

def findChessboardCorners(img, nx, ny):
    """
    Finds the chessboard corners of the supplied image (must be grayscale)
    nx and ny parameters respectively indicate the number of inner corners in the x and y directions
    """
    return cv2.findChessboardCorners(img, (nx, ny), None)

def showChessboardCorners(img, nx, ny, ret, corners):
    """
    Draws the chessboard corners of a given image
    nx and ny parameters respectively indicate the number of inner corners in the x and y directions
    ret and corners should represent the results from cv2.findChessboardCorners()
    """
    c_img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    # plt.axis('off')
    # plt.imshow(img)
    cv2.imshow('image', img)


def findImgObjPoints(imgs_paths, nx, ny):
    """
    Returns the objects and image points computed for a set of chessboard pictures taken from the same camera
    nx and ny parameters respectively indicate the number of inner corners in the x and y directions
    """
    objpts = []
    imgpts = []

    # Pre-compute what our object points in the real world should be (the z dimension is 0 as we assume a flat surface)
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    for img_path in imgs_paths:
        img = cv2.imread(img_path)
        img = cv2.resize(img, dsize=(640, 480), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = findChessboardCorners(gray, nx, ny)

        if ret:
            # Found the corners of an image
            imgpts.append(corners)
            # Add the same object point since they don't change in the real world
            objpts.append(objp)

    return objpts, imgpts

def undistort_image(img, objpts, imgpts):
    """
    Returns an undistorted image
    The desired object and image points must also be supplied to this function
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, gray.shape[::-1], None, None)
    print('un', img.shape)
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (img.shape[1], img.shape[0]), 1, (img.shape[1], img.shape[0]))
    # print('newcamera', newcameramtx)
    # cv2.imshow('roi', roi)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

def nothing(x):
    pass


calibration_dir = "camera_cal"
test_imgs_dir = "test_images"
output_imgs_dir = "output_images"
output_videos_dir = "output_videos"
cal_imgs_paths = glob.glob(calibration_dir + "/*.png")

cx = 7
cy = 5
opts, ipts = findImgObjPoints(cal_imgs_paths, cx, cy)
test_img = cv2.imread('frame1_9.jpg')
test_img_undist = undistort_image(test_img, opts, ipts)
cut_img = test_img_undist[0:145, :]
src_pts = np.array([[42, 145], [196, 0], [412, 0], [623, 145]], dtype=np.float32)
dst_pts = np.array([[0, 145], [0, 0], [640, 0], [640, 145]], dtype=np.float32)
m = cv2.getPerspectiveTransform(src_pts, dst_pts)
warped = cv2.warpPerspective(cut_img, m, (cut_img.shape[1], cut_img.shape[0]))  # (maxWidth, maxHeight))
hsl_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2HLS)
hsv_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)


black_win_name = 'black'
green_win_name = 'green'
green_hsv_win_name = 'green hsv'

windows_mode = cv2.WINDOW_NORMAL
cv2.imshow('origin', warped)
# blackeate window 'Settings'
cv2.namedWindow(black_win_name, windows_mode)
# create 6 trackbars for hsv-settings (black lane)
cv2.createTrackbar('h0', black_win_name, 0, 180, nothing)
cv2.createTrackbar('s0', black_win_name, 0, 255, nothing)
cv2.createTrackbar('l0', black_win_name, 0, 255, nothing)
cv2.createTrackbar('h1', black_win_name, 180, 180, nothing)
cv2.createTrackbar('s1', black_win_name, 255, 255, nothing)
cv2.createTrackbar('l1', black_win_name, 255, 255, nothing)

cv2.namedWindow(green_win_name, windows_mode)
# create 6 trackbars for hsv-settings (black lane)
cv2.createTrackbar('h0g', green_win_name, 0, 180, nothing)
cv2.createTrackbar('s0g', green_win_name, 0, 255, nothing)
cv2.createTrackbar('l0g', green_win_name, 0, 255, nothing)
cv2.createTrackbar('h1g', green_win_name, 180, 180, nothing)
cv2.createTrackbar('s1g', green_win_name, 255, 255, nothing)
cv2.createTrackbar('l1g', green_win_name, 255, 255, nothing)

cv2.namedWindow(green_hsv_win_name, windows_mode)
# create 6 trackbars for hsv-settings (black lane)
cv2.createTrackbar('h0gv', green_hsv_win_name, 0, 180, nothing)
cv2.createTrackbar('s0gv', green_hsv_win_name, 0, 255, nothing)
cv2.createTrackbar('l0gv', green_hsv_win_name, 0, 255, nothing)
cv2.createTrackbar('h1gv', green_hsv_win_name, 180, 180, nothing)
cv2.createTrackbar('s1gv', green_hsv_win_name, 255, 255, nothing)
cv2.createTrackbar('l1gv', green_hsv_win_name, 255, 255, nothing)

while (1):
    # get value of hsv-settings (black lane)
    h0 = cv2.getTrackbarPos('h0', black_win_name)
    s0 = cv2.getTrackbarPos('s0', black_win_name)
    l0 = cv2.getTrackbarPos('l0', black_win_name)
    h1 = cv2.getTrackbarPos('h1', black_win_name)
    s1 = cv2.getTrackbarPos('s1', black_win_name)
    l1 = cv2.getTrackbarPos('l1', black_win_name)

    # get value of hsv-settings (green lane)
    h0g = cv2.getTrackbarPos('h0g', green_win_name)
    s0g = cv2.getTrackbarPos('s0g', green_win_name)
    l0g = cv2.getTrackbarPos('l0g', green_win_name)
    h1g = cv2.getTrackbarPos('h1g', green_win_name)
    s1g = cv2.getTrackbarPos('s1g', green_win_name)
    l1g = cv2.getTrackbarPos('l1g', green_win_name)

    h0gv = cv2.getTrackbarPos('h0gv', green_hsv_win_name)
    s0gv = cv2.getTrackbarPos('s0gv', green_hsv_win_name)
    l0gv = cv2.getTrackbarPos('l0gv', green_hsv_win_name)
    h1gv = cv2.getTrackbarPos('h1gv', green_hsv_win_name)
    s1gv = cv2.getTrackbarPos('s1gv', green_hsv_win_name)
    l1gv = cv2.getTrackbarPos('l1gv', green_hsv_win_name)

    lower_black = np.array([h0, l0, s0])
    upper_black = np.array([h1, l1, s1])

    lower_green = np.array([h0g, l0g, s0g])
    upper_green = np.array([h1g, l1g, s1g])

    lower_green_v = np.array([h0gv, l0gv, s0gv])
    upper_green_v = np.array([h1gv, l1gv, s1gv])


    mask_black = cv2.inRange(hsl_warped, lower_black, upper_black)
    mask_green = cv2.inRange(hsl_warped, lower_green, upper_green)
    mask_green_v = cv2.inRange(hsv_warped, lower_green_v, upper_green_v)
    # print('mask', mask_green)
    frame = cv2.bitwise_or(mask_black, mask_green)
    cv2.imshow('green', mask_green)
    cv2.imshow('black', mask_black)
    cv2.imshow('test', frame)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()