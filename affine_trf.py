import cv2
import numpy as np
from PIL import Image
from shapely.geometry import LineString
from shapely.geometry import Point
# import math
from math import atan2, tan, degrees, sqrt, acos, cos, atan

def get_points(img):
    # Create dictionary for my_mouse func
    data = {}
    data['img'] = img.copy()
    data['points'] = []

    cv2.imshow('test', img)
    cv2.setMouseCallback('test', my_mouse_left_click, data)
    cv2.waitKey(0)
    # Convert array to np.array in shape n,2,2
    points = np.int16(data['points'])
    print('pnts from func', points)
    return points, data['img']

def my_mouse_left_click(event, x, y, flags, data):

    if event == cv2.EVENT_LBUTTONDOWN and len(data['points']) < 3:
        # print('CHECK', data['points'])
        data['points'].append((x,y))
        cv2.circle(data['img'], (x, y), 5, (255, 0, 255), thickness=-1)  # first point
        # print('2nd - check')
        cv2.imshow("test", data['img'])

    elif event == cv2.EVENT_LBUTTONDOWN and len(data['points']) < 4:
        data['points'].append((x, y))
        # data['points'].insert(0, [(x, y)])  # prepend the point
        cv2.circle(data['img'], (x, y), 5, (255, 0, 255), thickness=-1) # first point
        # print('3rd - check')
        cv2.imshow("test", data['img'])

def order_points(pts):
    pts = np.array(pts).astype(np.int)
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    # diff = np.diff(pts, axis=1)
    last = []
    for i in range(0, len(pts)):
        if pts[i][0] != rect[0][0] and pts[i][1] != rect[0][1]:
            if pts[i][0] != rect[2][0] and pts[i][1] != rect[2][1]:
                last.append(pts[i])
        else:
            continue
    if len(last) == 1:
        print('ERROR, last has 1 point')
    last = np.array(last, np.float32)

    last_s = last.sum(axis=1)
    rect[3] = last[np.argmin(last_s)]
    rect[1] = last[np.argmax(last_s)]

    print('rect', rect)

    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth + 1, 0],
        [maxWidth + 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))#(maxWidth, maxHeight))
    print('shape warped', warped.shape)
    # return the warped image
    return warped, rect

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

# Load BGR image
origin_f = cv2.imread('frame1_9.jpg', 1)
print(origin_f.shape)
origin = origin_f[(origin_f.shape[0]-500):(origin_f.shape[0]), 0:origin_f.shape[1]]
cv2.imshow('origin', origin)
# Load image in grayscale
image = cv2.imread("test_paper.jpg", 0)
image = image[(origin_f.shape[0]-500):(origin_f.shape[0]), 0:origin_f.shape[1]]
print(image.shape)
rows, cols, ch = origin.shape
print('rows', rows, 'colums', cols, 'channels', ch)
# Dilate the image to get rid of lines
dilate_im = cv2.dilate(image, np.ones((3, 3), np.uint8))
# Median blur - to get image with only shadows
blur_im = cv2.medianBlur(dilate_im, 11)
# Calculate the difference between original and obtained image. Reverse colours
diff_im = 255 - cv2.absdiff(image, blur_im)


draw_pnts, affine_img = get_points(diff_im)

warped_img, rect = four_point_transform(diff_im, draw_pnts)
# warped_gray_img = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)



canny_img   = cv2.Canny(diff_im, 20, 100)
cv2.imshow('canny', canny_img)
roi_vertices = [
    (0, diff_im.shape[0]),
    (0, 0),
#     (0, int(diff_im.shape[0]/2.5)),
#     (0+150, int(diff_im.shape[0]/8)),
#     (diff_im.shape[1]-130, int(diff_im.shape[0]/8)),
    (diff_im.shape[1], diff_im.shape[0]),
    (diff_im.shape[1], 0)
]

# roi_vertices = [
#     (0, diff_im.shape[0]),
#     (0, int(diff_im.shape[0]/2.5)),
#     (0+150, int(diff_im.shape[0]/8)),
#     (diff_im.shape[1]-130, int(diff_im.shape[0]/8)),
#     (diff_im.shape[1], int(diff_im.shape[0]/2.5)),
#     (diff_im.shape[1], diff_im.shape[0])
# ]


cv2.imshow('diff_im', diff_im)
'''
# Cropping to Region of Interest (ROI)
roi_image = region_of_interest(
    canny_img,
    np.array([roi_vertices], np.int32) # from list make numpy array
)

# array with lines [[x1, y1, x2, y2]]
lines = cv2.HoughLinesP(
    roi_image,
    rho=6,
    theta=np.pi/10,
    threshold=95,
    lines=np.array([]),
    minLineLength=10,
    maxLineGap=5
)


line_image = draw_lines(
    warped_img,
    lines,
    thickness=5
)
'''

# cv2.imshow('lined', line_image)
cv2.imshow('warped', warped_img)

# cv2.imshow('test', origin)
# cv2.imshow('res', af_img)
cv2.waitKey(0)

# cv2.imwrite('new_test_paper.png', warped_img)

cv2.destroyAllWindows()