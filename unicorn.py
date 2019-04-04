import cv2
import numpy as np
import math


def undistort_image(img, objpts, imgpts):
    """
    Returns an undistorted image
    The desired object and image points must also be supplied to this function
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, gray.shape[::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


'''
    Draw lines on image 
    !   Input image should be 3-channeled image
'''
def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
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


def region_of_interest(img, vertices):
    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(img)
    # Retrieve the number of color channels of the image.
    channel_count = img.shape[2]
    # Create a match color with the same color channel counts.
    match_mask_color = (255,) * channel_count

    # Fill inside the polygon
    cv2.fillPoly(mask, vertices, match_mask_color)

    # Returning the image only where mask pixels match
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image



'''
    Find low border of ROI (rectangular shape)
    
    Function has 2 options:
        show_border = 0     crop input image
        show_border = 1     draw the green line on input image
'''
def find_low_border_of_roi(img, roi_window, x_limit, show_border=0 ):
    crop_img = img.copy()
    height = img.shape[0]
    width = img.shape[1]

    line_not_found = 0  # boolean flag for checking condition of software

    # go through first [0] vertical line of image
    # search for white pixel
    for x in range(0, x_limit):
        for pix in range(height - 1, 0, -1):
            # check the left border pixel
            if img[pix][x][0] >= 254:
                for loc_x in range(width-1, width/2, -1):
                    for loc_pix in range(pix, pix - roi_window, -1):
                        # if white pixel is found
                        # crop/draw border
                        if img[loc_pix][loc_x][0] >= 254:
                            line_not_found = 1
                            if show_border == 0:
                                crop_img = img[0:loc_pix, 0:width]
                            else:
                                cv2.line(crop_img, (0, loc_pix), (width, loc_pix), color=(0, 255, 0), thickness=2)
                            break

                    if line_not_found == 1:
                        break
        if line_not_found == 1:
            break
    if line_not_found == 0:
        print("Ops! Border is not found")
    else:
        print("Border is found successfully!")

    return crop_img

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
            k_tg.append(round(k_temp, 4))
            k_rad.append(round(math.atan2(dy, dx), 4))
            k_deg.append(round(math.degrees(math.atan2(dy, dx)), 4))

    return k_tg, k_rad, k_deg


def get_k_b_line(line):
    k = round(float(line[3] - line[1])/(line[2] - line[0]), 4)
    b = round(line[1] - k * line[0], 4)

    # print(line[0], line[1], line[2], line[3], "k", k, "b", b)
    return k, b

btn_down = False

def get_points(img):
    # Create dictionary for my_mouse func
    data = {}
    data['img'] = img.copy()
    data['lines'] = []
    # Set the callback function for any mouse event
    cv2.imshow('test', img)
    cv2.setMouseCallback('test', my_mouse_left_click, data)
    cv2.waitKey(0)
    # Convert array to np.array in shape n,2,2
    points = np.uint16(data['lines'])
    return points, data['img']


def my_mouse_left_click(event, x, y, flags, data):
    global btn_down

    if event == cv2.EVENT_LBUTTONDOWN and btn_down:
        btn_down = False
        # add second point
        data['lines'][0].append((x, y))

        cv2.circle(data['img'], (x, y), 5, (255, 0, 255), thickness=-1)
        cv2.line(data['img'], data['lines'][0][0], data['lines'][0][1], (255, 0, 255), thickness=3)

        cv2.imshow('test', data['img'])

    elif event == cv2.EVENT_MOUSEMOVE and btn_down:
        # thi is just for a ine visualization
        image = data['img'].copy()
        cv2.line(image, data['lines'][0][0], (x, y), (255, 0, 255), 3)
        cv2.imshow("test", image)

    elif event == cv2.EVENT_LBUTTONDOWN and len(data['lines']) < 3:
        btn_down = True
        data['lines'].insert(0, [(x, y)])  # prepend the point
        cv2.circle(data['img'], (x, y), 5, (255, 0, 255), thickness=-1) # first point
        cv2.imshow("test", data['img'])
