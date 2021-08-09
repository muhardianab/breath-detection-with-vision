import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

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
    ret, frame2 = cap.read()
    frame2 = rescale_frame(frame2, percent=rescale_percent)
    
    # Detect ROI
    next_image = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    nose = noseCascade.detectMultiScale(next_image, scaleFactor=5, minNeighbors=1)
    #scaleFactor value define range between camera and the object
    #minNeighbors specifying how many neighbors each candidate rectangle should have before define the object should

    # Optical Flow - Dense / Farneback
    flow = cv2.calcOpticalFlowFarneback(prev_image, next_image, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Draw a rectangle around the body
    for (x, y, w, h) in nose:
        roi_gray = next_image[y:y+h, x:x+w]   #(ycoord-start, ycoord-end)
        #roi_color = bgr[y:y+h, x:x+w]

    # save frame last detection
        img_item = "image.png"
        cv2.imwrite(img_item, bgr) #(img_item, roi)

    # labeled object while recognize
        color = (255, 0, 0)     #BGR 0-255
        stroke = 2
        weight = x + w      #end coord x
        height = y + h      #end coord y
        cv2.rectangle(bgr, (x, y), (weight, height), color, stroke)

    rgb_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # show window frame
    cv2.imshow('frame', rgb_frame)
    k = cv2.waitKey(20) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png', frame2)
        cv2.imwrite('opticalhsv.png', bgr)
    prev_image = next_image

cap.release()
cv2.destroyAllWindows()