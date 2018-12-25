import cv2
import numpy as np

def undistort_image(img, objpts, imgpts):
    """
    Returns an undistorted image
    The desired object and image points must also be supplied to this function
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, gray.shape[::-1], None, None)
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

def bird_eye_view(img):
    pnt1 = [264, 0]
    pnt2 = [390, 0]
    pnt3 = [640, 228]
    pnt4 = [161, 228]
    src_pts = np.array([pnt1, pnt2, pnt3, pnt4], dtype=np.float32)
    dst_pts = np.array([[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, m, (img.shape[1], img.shape[0]))  # (maxWidth, maxHeight))
    return warped

def hsv_green_black_mask(bgr_img):
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

    h0 = 0
    h1 = 21
    s0 = 14
    s1 = 78
    v0 = 29
    v1 = 103
    lower_black = np.array([h0, s0, v0])
    upper_black = np.array([h1, s1, v1])

    h0g = 58
    h1g = 86
    s0g = 31
    s1g = 122
    v0g = 114
    v1g = 150
    lower_green = np.array([h0g, s0g, v0g])
    upper_green = np.array([h1g, s1g, v1g])

    mask_black = cv2.inRange(hsv_img, lower_black, upper_black)
    mask_green = cv2.inRange(hsv_img, lower_green, upper_green)

    img_hlv_black_green_bin = cv2.bitwise_or(mask_black, mask_green)

    return img_hlv_black_green_bin


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    img_sx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    img_sy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    img_s = np.sqrt(img_sx ** 2 + img_sy ** 2)
    img_s = np.uint8(img_s * 255 / np.max(img_s))
    binary_output = 0 * img_s
    binary_output[(img_s >= thresh[0]) & (img_s <= thresh[1])] = 1
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Calculate gradient direction
    # Apply threshold
    img_sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    img_sy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    grad_s = np.arctan2(np.absolute(img_sy), np.absolute(img_sx))

    binary_output = 0 * grad_s  # Remove this line
    binary_output[(grad_s >= thresh[0]) & (grad_s <= thresh[1])] = 1
    return binary_output


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    if orient == 'x':
        img_s = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    else:
        img_s = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    img_abs = np.absolute(img_s)
    img_sobel = np.uint8(255 * img_abs / np.max(img_abs))

    binary_output = 0 * img_sobel
    binary_output[(img_sobel >= thresh[0]) & (img_sobel <= thresh[1])] = 1
    return binary_output

def moving_average(a, n=3):
    # Moving average
    a = np.array(a).astype(np.int)
    ret = np.cumsum(a, dtype=float)
    ret = ret
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n