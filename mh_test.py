#!/usr/bin/python

'''
* assumption #1 : lane is oriented to be parallel with the direction of movement
*                 it means that lines will be extended slightly inwards
* assumption #2 : lines will never reach the horizon, either disappearing with distance
'''

import cv2
import numpy as np

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

# reading in an image
image = cv2.imread("frame1.png")
# reseizing the image
resize_koef = 0.8 # for sample1
# resize_koef = 1     # for frame6
new_width   = int(image.shape[1] * resize_koef)
new_heigth  = int(image.shape[0] * resize_koef)
resized_img = cv2.resize(image, (new_width, new_heigth))
# crop the resized image (y, x)
# cropped_img = resized_img[0:360,:] # for sample1
# cropped_img = resized_img[0:170,:]
cropped_img = resized_img
gray_roi    = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
canny_img   = cv2.Canny(gray_roi, 100, 400)
# info about images
print('This image is:',     type(image),        'with dimensions:', image.shape)
print('Reseized image is:', type(resized_img),  'with dimensions:', resized_img.shape)
print('Cropped image is:',  type(cropped_img),  'with dimensions:', cropped_img.shape)
# define the Region of Interest (ROI)
# point (0, 0) - upper left corner!
''' for sample1
center_roi_vertice = (int(cropped_img.shape[1]/1.7), int(cropped_img.shape[0]/2.7)) # for sample1
roi_vertices = [
    (0, cropped_img.shape[0]),
    center_roi_vertice,
    (cropped_img.shape[1], cropped_img.shape[0])
]

'''
# for frame 6 & frame 1
roi_vertices = [
    (0+15+50, cropped_img.shape[0]),
    (0+130+50, int(cropped_img.shape[0]/3.3)),
    (cropped_img.shape[1]-110, int(cropped_img.shape[0]/3.3)),
    (cropped_img.shape[1], cropped_img.shape[0])
]
'''
cv2.circle(cropped_img, roi_vertices[0], 5, (0, 255, 0), thickness=-1)
cv2.circle(cropped_img, roi_vertices[1], 5, (0, 255, 0), thickness=-1)
cv2.circle(cropped_img, roi_vertices[2], 5, (0, 255, 0), thickness=-1)
cv2.circle(cropped_img, roi_vertices[3], 5, (0, 255, 0), thickness=-1)
'''
# cv2.imshow('original', image)
# cv2.imshow('resized', resized_img)
# cv2.imshow('cropped', cropped_img)
cv2.imshow('canny', canny_img)
cv2.imwrite('canny_frame1.png', canny_img)

# Cropping to Region of Interest (ROI)
roi_image = region_of_interest(
    canny_img,
    np.array([roi_vertices], np.int32) # from list make numpy array
)

# cv2.imshow('old_roi', roi_image)

# array with lines [[x1, y1, x2, y2]]
lines = cv2.HoughLinesP(
    roi_image,
    rho=6,
    theta=np.pi / 60,
    threshold=160,
    lines=np.array([]),
    minLineLength=40,
    maxLineGap=25
)

''' analog of draw_lines
for i in lines:
    print('1 ', i)
    print('0 ', i[0])
    print('00 ', i[0][0])

    cv2.line(cropped_img, (i[0][0], i[0][1]),(i[0][2], i[0][3]), (0, 0, 255), thickness=3)
'''
line_image = draw_lines(
    cropped_img,
    lines,
    thickness=5
)

cv2.imshow('with_lines', line_image)
'''
# Let's divide lines into left_line and right_line
left_line_x = []
left_line_y = []

right_line_x = []
right_line_y = []

for line in lines:
    for x1, y1, x2, y2 in line:
        slope = float(y2 - y1)/float(x2 - x1) # <-- Calculating the slope.
        #print(line)
        #print('Slope ', slope )
        if math.fabs(slope) < 0.5: # <-- Only consider extreme slope
            continue
        if slope <= 0: # <-- If the slope is negative, left group.
            left_line_x.extend([x1, x2])
            left_line_y.extend([y1, y2])
        else: # <-- Otherwise, right group.
            right_line_x.extend([x1, x2])
            right_line_y.extend([y1, y2])
cv2.imshow('cropped', cropped_img)
min_y = cropped_img.shape[0] * 0.4  # <-- Just below the horizon
max_y = cropped_img.shape[0]        # <-- The bottom of the image

print('min_y', min_y)
min_y = int(min_y)
poly_left = np.poly1d(np.polyfit(
    left_line_y,
    left_line_x,
    deg=1
))

left_x_start = int(poly_left(max_y))
left_x_end = int(poly_left(min_y))

poly_right = np.poly1d(np.polyfit(
    right_line_y,
    right_line_x,
    deg=1
))


right_x_start = int(poly_right(max_y))
right_x_end = int(poly_right(min_y))


line_image = draw_lines(
    cropped_img,
    [[
        [left_x_start, max_y, left_x_end, min_y],
        [right_x_start, max_y, right_x_end, min_y],
    ]],
    thickness=5
)

cv2.imwrite('new_frame1.png', line_image)

cv2.imshow('ROI',    roi_image)
cv2.imshow('Origin', line_image)
'''

cv2.waitKey(0)
cv2.destroyAllWindows()



