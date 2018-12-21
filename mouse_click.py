#!/usr/bin/python

import cv2
import numpy as np

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
    print("PPPP", points)
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


origin = cv2.imread('frame1.png', 1)
cv2.namedWindow('test', cv2.WINDOW_AUTOSIZE)
pts, final_img = get_points(origin)


cv2.imshow('test', final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

