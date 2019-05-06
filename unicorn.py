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

# Need comments
# Fix name of pnt1- 4 => lt, lb etc.
def bird_eye_view(img, pnt1 = [264, 0], pnt2 = [390, 0], pnt3 = [640, 228], pnt4 = [161, 228]):
    # pnt1 = [264, 0]
    # pnt2 = [390, 0]
    # pnt3 = [640, 228]
    # pnt4 = [161, 228]
    src_pts = np.array([pnt1, pnt2, pnt3, pnt4], dtype=np.float32)
    dst_pts = np.array([[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, m, (img.shape[1], img.shape[0]))  # (maxWidth, maxHeight))
    return warped

def undo_bird_eye_view(img, lt_pnt = [264, 0], rt_pnt = [390, 0], rb_pnt = [640, 228], lb_pnt = [161, 228],
                            dst_lt_pnt = [0, 0], dst_rt_pnt=[640, 0], dst_rb_pnt=[640, 228], dst_lb_pnt=[0, 228]):
    # pnt1 = [264, 0]
    # pnt2 = [390, 0]
    # pnt3 = [640, 228]
    # pnt4 = [161, 228]
    src_pts = np.array([lt_pnt, rt_pnt, rb_pnt, lb_pnt], dtype=np.float32)
    dst_pts = np.array([dst_lt_pnt, dst_rt_pnt, dst_rb_pnt, dst_lb_pnt], dtype=np.float32)
    m = cv2.getPerspectiveTransform(src_pts, dst_pts)
    unwarped = cv2.warpPerspective(img, m, (img.shape[1], img.shape[0]))  # (maxWidth, maxHeight))
    return unwarped


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

'''
    Crop input image to get prepared image 
    for perspective transformation
    picture should have at least two lines 
    edges of lines can not be at the edge of image 
'''
def find_low_border_of_roi_m2(img, y_limit, x_limit, show_border=0 ):
    crop_img = img.copy()
    height = img.shape[0]
    width = img.shape[1]
    left_right_x_lim = 15
    line_not_found = 0  # boolean flag for checking condition of software
    # go through first [0] vertical line of image
    # search for white pixel
    for x in range(0, x_limit):
        for y in range(height - 1, 0, -1):
            # check the left border pixel
            if img[y][x][0] >= 254:
                if x > left_right_x_lim:
                    for loc_x in range(width-1, 0, -1):
                        for loc_pix in range(y, y - y_limit, -1):
                            # if white pixel is found
                            # crop/draw border
                            if img[loc_pix][loc_x][0] >= 254:
                                # crop the image only if pixel is not on the
                                # edge of image
                                if loc_x < width - 1 - left_right_x_lim:
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
        print("Oops! Border is not found")
    else:
        print("Border is found successfully!")

    return crop_img



# def divide_height_image(img, parts):
#     height = img.shape[0]
#     if height%parts == 0:
#         dh = height / parts
#     else:
#         dh = height / parts
#         last_dh = height - (dh * parts)




    '''
    Get image with all black pixels 
    size is simillar as input image
'''
def make_image_all_black(img):
    black_img = np.copy(img)
    for y in range(0, black_img.shape[0]):
        for x in range(0, black_img.shape[1]):
            black_img[y][x] = [0, 0, 0]
    return black_img


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

def get_k_deg(line):
    dx = line[2] - line[0]
    dy = line[3] - line[1]

    if dx == 0:
        k_deg = 0
    else:
        # k_tg = round(round(dy / float(dx), 4), 4)
        k_deg = round(math.degrees(math.atan2(dy, dx)), 1)

    if k_deg > 90:
        k_deg = -(180 - k_deg)
    elif k_deg < -90:
        k_deg = -(-180 - k_deg)

    return k_deg

def get_pnts_of_line(img, color):
    pnts_x = []
    pnts_y = []
    for y in range(0, img.shape[0]):
        for x in range(0, img.shape[1]):
            if img[y][x][0] == color[0] and img[y][x][1] == color[1] and img[y][x][2] == color[2]:
                pnts_x.append(x)
                pnts_y.append(y)

    y_min = min(pnts_y)
    y_max = max(pnts_y)

    high_x = []
    low_x = []
    for x in range(0, img.shape[1]):
        if img[y_min][x][0] == color[0] and img[y_min][x][1] == color[1] and img[y_min][x][2] == color[2]:
            high_x.append(x)

        if img[y_max][x][0] == color[0] and img[y_max][x][1] == color[1] and img[y_max][x][2] == color[2]:
            low_x.append(x)

    high_x_val = int(np.mean(high_x))
    low_x_val = int(np.mean(low_x))

    line = [high_x_val, y_min, low_x_val, y_max]
    return line





def get_k_b_line(line):
    k = round(float(line[3] - line[1])/(line[2] - line[0]), 4)
    b = round(line[1] - k * line[0], 4)

    # print(line[0], line[1], line[2], line[3], "k", k, "b", b)
    return k, b

'''
    Get 4 coordinates of corner points 
    These coordinates are needed for bird_eye_view function
    left bottom point and right bottom point
    are got just as first and the last point in the line 
'''
def get_4_pnts_for_warping(img, top_dx = 12, draw_pnts = 0):
    height = img.shape[0]
    width = img.shape[1]

    rb_pnt = [width - 1, height - 1]
    lb_pnt = [0, height - 1]

    # rb_pnt = [width - 10, height - 1]
    # lb_pnt = [0 + 10, height - 1]


    # left bottom corner point
    # for w in range(0, width):
    #     if img[height - 1][w][0] >= 254:
    #         lb_pnt = [w, height - 1]
    #         break

    # right bottom corner point
    # for w in range(width - 1, 0, -1):
    #     if img[height - 1][w][0] >= 254:
    #         rb_pnt = [w, height - 1]
    #         break

    next_y = 0
    lt_found = 0
    # left top corner point
    for y in range(0, height):
        next_y = 0
        for top in range(0, width):
            if img[y][top][0] > 0:
                if top > 135: # think about how to figure out limits automatically
                    next_y = 1
                else:
                    lt_pnt = [top - top_dx, y]
                    lt_found = 1
                    break

            if next_y == 1 or lt_found == 1:
                break
        if lt_found == 1:
            break

    rt_found = 0
    check_r_line = 0
    check_next_y = 0
    # right top corner point
    for y in range(0, height):
        next_y = 0
        for x in range(width - 1, 0, -1):
            if img[y][x][0] > 0:
                if x < 170:
                    next_y = 1
                else:
                    # for check_y in range(y, y + 10):
                    #     check_next_y = 0
                    #     for check_x in range(x, width):
                    #         if img[check_y][check_x][0] > 0:
                    #             check_r_line += 1
                    #             check_next_y = 1
                    #             break
                    #         if check_next_y == 1:
                    #             break
                    # print 'check right line', check_r_line
                    # if check_r_line >= 10 and check_next_y == 1:
                    rt_pnt = [x + top_dx, 0]
                    rt_found = 1
                    break
                    # else:
                    #     rt_found = 0
                    #     next_y = 1

            if next_y == 1 or rt_found == 1:
                break
        if rt_found == 1:
            break

    if draw_pnts == 1:
        cv2.circle(img, (lb_pnt[0], lb_pnt[1]), 2, (255, 0, 0), thickness=-1)
        cv2.circle(img, (rb_pnt[0], rb_pnt[1]), 2, (255, 0, 0), thickness=-1)
        cv2.circle(img, (lt_pnt[0], lt_pnt[1]), 2, (255, 0, 0), thickness=-1)
        cv2.circle(img, (rt_pnt[0], rt_pnt[1]), 2, (255, 0, 0), thickness=-1)

    print 'lt', lt_pnt, 'rt', rt_pnt

    return lb_pnt, rb_pnt, lt_pnt, rt_pnt

# fix deathzone -> arg
def count_number_of_extrema(arr, deathzone = 0.3):
    extrema_count = 0
    for m in range(1, len(arr) - 1):
        if arr[m] > deathzone:
            if arr[m] > arr[m - 1] and arr[m] > arr[m + 1]:
                extrema_count += 1
            if arr[m] < arr[m - 1] and arr[m] < arr[m + 1]:
                extrema_count += 1

    return extrema_count

'''
@brief      Get array of mean values of pixel intensity 
@param      gray image (numpy array)
@return     list of mean values of pixel intensity 
'''
def get_mean_intensity(gray_image):

    height = gray_image.shape[0]
    width = gray_image.shape[1]
    indexes = []
    mean_lines = []
    temp_line = []
    for x in range(0, width):
        for y in range(0, height - 1):
            temp_line.append(gray_image[y][x])

        mean_lane = np.mean(temp_line)
        temp_line[:] = []
        mean_lines.append(mean_lane)
        indexes.append(x)
    # to normalize limits of y
    max_mean = max(mean_lines)
    norm_k = round(height / max_mean, 4)
    for m in range(0, len(mean_lines)):
        mean_lines[m] = mean_lines[m] * norm_k

    return mean_lines, indexes

'''
@brief      Get array of boundary values of pixel intensity 
@note       upper limit is the max value of mean values
@param      mean_val    -  array of mean values
            threshold   -  value for 'step' response
@return     list of boundary values of mean values 
'''
def get_boundaries_for_intensity(mean_val, threshold):
    boundaries = []
    for mean in range(0, len(mean_val)):
        if mean_val[mean] > threshold:
            boundaries.append(max(mean_val))
        else:
            boundaries.append(0)

    return boundaries



def get_only_pick_pnts(boundaries, mean_arr, start_val, mean_indx):
    pick_pnts = []
    pick_indexes = []
    end_of_peak = 0
    for b in range(start_val, len(boundaries) - 1):
        if boundaries[b] == 0 and boundaries[b + 1] > 0:
            if end_of_peak == 0:    # maybe it is always true
                pick_pnts.append(mean_arr[b + 1])
                pick_indexes.append(mean_indx[b+1])
        if boundaries[b] > 0 and boundaries[b + 1] > 0 and boundaries[b] == boundaries[b + 1]:
            if end_of_peak == 0:    # maybe it is always true
                pick_pnts.append(mean_arr[b + 1])
                pick_indexes.append(mean_indx[b + 1])

        if boundaries[b] > 0 and boundaries[b + 1] == 0:
            end_of_peak = b + 1
            break

    return pick_pnts, end_of_peak, pick_indexes

# fix global extrema, calculate from whole picture
def get_picks_arrays(extrema_num, mean_arr, boundary_arr, indexes, img, global_extrema=3):
    # global_extrema = count_number_of_extrema
    width = img.shape[1]
    left_pick = []
    center_pick = []
    right_pick = []

    left_x = []
    center_x = []
    right_x = []
    if extrema_num == 1:
        if global_extrema == 3:
            temp_pick, temp_end, temp_x = get_only_pick_pnts(boundary_arr, mean_arr, 0, indexes)
            if max(temp_x) < width/3:
                left_pick = temp_pick
                end_of_left_pick = temp_end
                left_x = temp_x
            elif max(temp_x) > (width - 2 * width/3) and max(temp_x) < (width - width/3):
                center_pick = temp_pick
                end_of_center_pick = temp_end
                center_x = temp_x
            elif max(temp_x) > (width - width/3):
                right_pick = temp_pick
                end_of_right_pick = temp_end
                right_x = temp_x

        # left_pick, end_of_left_pick, left_x = get_only_pick_pnts(boundary_arr, mean_arr, 0, indexes)
        print("One pick detected!")

    elif extrema_num == 2:
        if global_extrema == 3:
            temp_pick, temp_end, temp_x = get_only_pick_pnts(boundary_arr, mean_arr, 0, indexes)
            # print("max x", temp_x, width/3, width - 2 * width/3, width - width/3)
            if max(temp_x) < width/3:
                left_pick = temp_pick
                end_of_left_pick = temp_end
                left_x = temp_x
            elif max(temp_x) > (width - 2 * width/3) and max(temp_x) < (width - width/3):
                center_pick = temp_pick
                end_of_center_pick = temp_end
                center_x = temp_x
            elif max(temp_x) > (width - width/3):
                right_pick = temp_pick
                end_of_right_pick = temp_end
                right_x = temp_x

            temp_pick, temp_end, temp_x = get_only_pick_pnts(boundary_arr, mean_arr, temp_end, indexes)
            if max(temp_x) < width/3:
                left_pick = temp_pick
                end_of_left_pick = temp_end
                left_x = temp_x
            elif max(temp_x) > (width - 2 * width/3) and max(temp_x) < (width - width/3):
                center_pick = temp_pick
                end_of_center_pick = temp_end
                center_x = temp_x
            elif max(temp_x) > (width - width/3):
                right_pick = temp_pick
                end_of_right_pick = temp_end
                right_x = temp_x

        # left_pick, end_of_left_pick, left_x = get_only_pick_pnts(boundary_arr, mean_arr, 0, indexes)
        # center_pick, end_of_center_pick, center_x = get_only_pick_pnts(boundary_arr, mean_arr, end_of_left_pick, indexes)
        print("Two picks detected!")

    elif extrema_num == 3:

        left_pick, end_of_left_pick, left_x  = get_only_pick_pnts(boundary_arr, mean_arr, 0, indexes)
        center_pick, end_of_center_pick, center_x = get_only_pick_pnts(boundary_arr, mean_arr, end_of_left_pick, indexes)
        right_pick, end_of_right_pick, right_x = get_only_pick_pnts(boundary_arr, mean_arr, end_of_center_pick, indexes)
        print("Three picks detected!")

    return left_pick, center_pick, right_pick, left_x, center_x, right_x

def get_picks_coordinates(img, left_pick_arr, left_x, center_pick_arr, center_x, right_pick_arr, right_x):

    height = img.shape[0]
    if len(left_pick_arr) > 0:
        left_pick_y = max(left_pick_arr)
        left_indx = left_pick_arr.index(left_pick_y)
        left_pick_y = height - left_pick_y # because top left corner is 0 for y
        left_pick_x = left_x[left_indx]
        left_pick_coo = [int(left_pick_x), int(left_pick_y)]
    else:
        print("No left pick")
        left_pick_coo = []

    if len(center_pick_arr) > 0:
        center_pick_y = max(center_pick_arr)
        center_indx = center_pick_arr.index(center_pick_y)
        center_pick_y = height - center_pick_y
        center_pick_x = center_x[center_indx]
        center_pick_coo = [int(center_pick_x), int(center_pick_y)]
    else:
        print("No center pick")
        center_pick_coo = []

    if len(right_pick_arr) > 0:
        right_pick_y = max(right_pick_arr)
        right_indx = right_pick_arr.index(right_pick_y)
        right_pick_y = height - right_pick_y
        right_pick_x = right_x[right_indx]
        right_pick_coo = [int(right_pick_x), int(right_pick_y)]
    else:
        print("No right pick")
        right_pick_coo = []

    return left_pick_coo, center_pick_coo, right_pick_coo

def draw_pick_pnts(img, left, center, right):
    if len(left) > 0:
        cv2.circle(img, (left[0], left[1]), 3, (255, 0, 0), thickness=-1)
    if len(center) > 0:
        cv2.circle(img, (center[0], center[1]), 3, (0, 255, 0), thickness=-1)
    if len(right) > 0:
        cv2.circle(img, (right[0], right[1]), 3, (0, 0, 255), thickness=-1)

def draw_pick_mask(img, start_x, end_x, pick_pnt, color=[255, 100, 100]):
    height = img.shape[0]
    if len(pick_pnt) > 0:
        for y in range(0, height):
            for x in range(start_x, end_x):
                img[y][x] = color

            # for x in range(pick_pnt[0], pick_pnt[0] + line_width/2):
            #     img[y][x] = color
    else:
        print("No pick to draw mask")

# def process_part_of_img(part_img, bound_threshold, line_width):
def draw_magic(part_img, bound_threshold, line_width):
    mean, mean_indx = get_mean_intensity(part_img)
    bound = get_boundaries_for_intensity(mean, bound_threshold)
    extrema = count_number_of_extrema(mean)
    left_pick, center_pick, right_pick, left_x, center_x, right_x = get_picks_arrays(extrema, mean, bound, mean_indx, part_img)
    left_pick_coo, center_pick_coo, right_pick_coo = get_picks_coordinates(part_img, left_pick, left_x, center_pick,
                                                                           center_x, right_pick, right_x)

    if len(left_x) > 0:
        end_left_x = max(left_x)
        start_left_x = min(left_x)
    else:
        end_left_x = start_left_x = -1
    if len(center_x) > 0:
        end_center_x = max(center_x)
        start_center_x = min(center_x)
    else:
        end_center_x = start_center_x = -1
    if len(right_x) > 0:
        end_right_x = max(right_x)
        start_right_x = min(right_x)
    else:
        end_right_x = start_right_x = -1

    # print('left', start_left_x, end_left_x, 'center', start_center_x, end_center_x,
    #       'right', start_right_x, end_right_x)


    draw_pick_mask(part_img, start_left_x, end_left_x, left_pick_coo)
    draw_pick_mask(part_img, start_center_x, end_center_x, center_pick_coo)
    draw_pick_mask(part_img, start_right_x, end_right_x, right_pick_coo)

def get_mask_data(part_img, bound_threshold, line_width):
    mean, mean_indx = get_mean_intensity(part_img)
    bound = get_boundaries_for_intensity(mean, bound_threshold)
    extrema = count_number_of_extrema(mean)
    left_pick, center_pick, right_pick, left_x, center_x, right_x = get_picks_arrays(extrema, mean, bound, mean_indx, part_img)
    left_pick_coo, center_pick_coo, right_pick_coo = get_picks_coordinates(part_img, left_pick, left_x, center_pick,
                                                                           center_x, right_pick, right_x)

    if len(left_x) > 0:
        end_left_x = max(left_x)
        start_left_x = min(left_x)
    else:
        end_left_x = start_left_x = -1
    if len(center_x) > 0:
        end_center_x = max(center_x)
        start_center_x = min(center_x)
    else:
        end_center_x = start_center_x = -1
    if len(right_x) > 0:
        end_right_x = max(right_x)
        start_right_x = min(right_x)
    else:
        end_right_x = start_right_x = -1



    left_part_mask = np.copy(part_img)
    left_height = left_part_mask.shape[0]
    left_width = left_part_mask.shape[1]
    if len(left_pick_coo) > 0:
        for y in range(0, left_height):
            # print("y", y, 'left star/end', start_left_x, end_left_x)
            for x in range(start_left_x, end_left_x+1):
                # print('x', x)
                left_part_mask[y][x] = [255, 255, 255]

            for x in range(0, start_left_x):
                # print('clear x', x)
                left_part_mask[y][x] = [0, 0, 0]

            # for x in range(left_pick_coo[0], left_pick_coo[0] + line_width/2):
            #     left_part_mask[y][x] = [255, 255, 255]

            for x in range(end_left_x+1, part_img.shape[1]):
                # print('clear x', x)
                left_part_mask[y][x] = [0, 0, 0]
    else:
        print("No Left pick to get mask")
        for y in range(0, left_height):
            for x in range(0, left_width):
                left_part_mask[y][x] = [0, 0, 0]

    center_part_mask = np.copy(part_img)
    center_height = center_part_mask.shape[0]
    center_width = center_part_mask.shape[1]
    if len(center_pick_coo) > 0:
        for y in range(0, center_height):
            # print("y", y, 'center star/end', start_center_x, end_center_x)
            for x in range(start_center_x, end_center_x+1):
                center_part_mask[y][x] = [255, 255, 255]

            # for x in range(center_pick_coo[0], center_pick_coo[0] + line_width / 2):
            #     center_part_mask[y][x] = [255, 255, 255]
            # draw all black from left side
            for x in range(0, start_center_x):
                # print('x', x)

                center_part_mask[y][x] = [0, 0, 0]
            # draw all black from right side
            for x in range(end_center_x+1, part_img.shape[1]):
                # print('x', x)

                center_part_mask[y][x] = [0, 0, 0]
    else:
        print("No Center pick to get mask")
        for y in range(0, center_height):
            for x in range(0, center_width):
                center_part_mask[y][x] = [0, 0, 0]

    right_part_mask = np.copy(part_img)
    right_height = right_part_mask.shape[0]
    right_width = right_part_mask.shape[1]
    if len(right_pick_coo) > 0:
        for y in range(0, right_height):
            # print("y", y, 'right star/end', start_right_x, end_right_x)
            for x in range(start_right_x, end_right_x+1):
                right_part_mask[y][x] = [255, 255, 255]

            # for x in range(right_pick_coo[0], right_pick_coo[0] + line_width / 2):
            #     right_part_mask[y][x] = [255, 255, 255]

            # draw all black from right side
            for x in range(end_right_x+1, part_img.shape[1]):
                right_part_mask[y][x] = [0, 0, 0]

            # draw all black from left side
            for x in range(0, start_right_x):
                right_part_mask[y][x] = [0, 0, 0]
    else:
        print("No Right pick to get mask")
        for y in range(0, right_height):
            for x in range(0, right_width):
                right_part_mask[y][x] = [0, 0, 0]

    return left_part_mask, center_part_mask, right_part_mask


def draw_pw_lines(img,pts,color):
    # draw lines
    pts = np.int_(pts)
    for i in range(10):
        x1 = pts[0][i][0]
        y1 = pts[0][i][1]
        x2 = pts[0][i+1][0]
        y2 = pts[0][i+1][1]
        cv2.line(img, (x1, y1), (x2, y2),color,5)

def draw_magic_line(img, pnts, color, thickness=5):
    pnts = np.int_(pnts)
    cv2.line(img, (pnts[0][0][0], pnts[0][0][1]), (pnts[0][1][0], pnts[0][1][1]), color, thickness=thickness)



# need to fix for 2 (1) line
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
