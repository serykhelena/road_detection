from __future__ import print_function
import numpy as np
import cv2
import glob
import pickle

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




calibration_dir     = "camera_cal"
# Let's get all our calibration image paths
cal_imgs_paths = glob.glob(calibration_dir + "/*.png")
cx = 7
cy = 5
# ret, corners = findChessboardCorners(gray_img_rescaled, cx, cy)
# showChessboardCorners(img, cx, cy, ret, corners)
opts, ipts = findImgObjPoints(cal_imgs_paths, cx, cy)

with open('cal_param_opts.txt', 'wb') as fp:
    pickle.dump(opts, fp)

with open('cal_param_ipts.txt', 'wb') as fp:
    pickle.dump(ipts, fp)

