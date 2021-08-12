import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from numpy.lib.shape_base import row_stack

rescale_percent = 75

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_NEAREST) #INTER_NEAREST, INTER_AREA

# function for change resolution
def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)

# Create the haar cascade
#fullbodyCascade = cv2.CascadeClassifier('/home/muhardianab/opencv/data/haarcascades/haarcascade_fullbody.xml')
noseCascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')

# Capture frame as video stream
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(cv2.samples.findFile("videoplayback-faces.mp4"))

# make prev_image before processed to detect
ret, frame1 = cap.read()
frame1 = rescale_frame(frame1, percent=rescale_percent)
prev_image = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = rescale_frame(frame, percent=rescale_percent)
    
    # Detect ROI
    next_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    nose = noseCascade.detectMultiScale(next_image, scaleFactor=5, minNeighbors=1)

    # Optical Flow - Dense / Farneback
    flow = cv2.calcOpticalFlowFarneback(prev_image, next_image, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Draw a rectangle around the body
    for (x, y, w, h) in nose:
        roi_gray = next_image[y:y+h, x:x+w]   #(ycoord-start, ycoord-end)
        #roi_color = next_image[y:y+h, x:x+w]

    # save frame last detection
        cv2.imwrite("image.png", roi_gray) #(img_item, roi)
        roi_color = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)
        cv2.imwrite("image2.png", roi_color)
        np.savetxt("test1.txt", roi_gray, fmt='%.5e', delimiter=',')

    # labeled object while recognize
        color = (255, 0, 0)     #BGR 0-255
        stroke = 2
        weight = x + w      #end coord x
        height = y + h      #end coord y
        cv2.rectangle(bgr, (x, y), (weight, height), color, stroke)

    rgb_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    # show window frame
    both = cv2.hconcat([frame, rgb_frame])
    cv2.imshow('cam', both)
    cv2.imshow('prev_image', prev_image)
    
    # button
    k = cv2.waitKey(20) & 0xFF
    if k == ord('q'):                       # press q to quit
        break
    elif k == ord('s'):                     # press s to write image
        cv2.imwrite('opticalfb.png', frame)
        cv2.imwrite('opticalhsv.png', bgr)
    prev_image = next_image

cap.release()
cv2.destroyAllWindows()