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

# Capture frame as video stream
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(cv2.samples.findFile("videoplayback-faces.mp4"))

# Create the haar cascade
#fullbodyCascade = cv2.CascadeClassifier('/home/muhardianab/opencv/data/haarcascades/haarcascade_fullbody.xml')
noseCascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')

roi_temp = None
while True:
    ret, frame_temp = cap.read()
    frame_temp = rescale_frame(frame_temp, percent=rescale_percent)
    
    # Detect ROI
    frame_temp = cv2.cvtColor(frame_temp, cv2.COLOR_BGR2GRAY)
    nose = noseCascade.detectMultiScale(frame_temp, scaleFactor=5, minNeighbors=1)
    for (x, y, w, h) in nose:
        roi_temp = frame_temp[y:y+h, x:x+w]
    print(roi_temp, "===========not detected")
    cv2.imwrite('frame_temp.jpg', frame_temp)
    cv2.imshow('temp', frame_temp)
    if roi_temp is None:
        continue
    else:
        cv2.imwrite('roi_temp.jpg', roi_temp)
        print(roi_temp, "===========detected")
        break

# make prev_image before processed to detect
#ret, frame1 = cap.read()
#frame1 = rescale_frame(frame1, percent=rescale_percent)
#prev_image = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
#frame1 = np.loadtxt("kosong.txt", delimiter=',')
hsv = np.zeros_like(roi_temp)       #prev_image = roi_temp
hsv[..., 1] = 255

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = rescale_frame(frame, percent=rescale_percent)
    
    # Detect ROI
    next_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    nose = noseCascade.detectMultiScale(next_image, scaleFactor=5, minNeighbors=1)
    #scaleFactor value define range between camera and the object
    #minNeighbors specifying how many neighbors each candidate rectangle should have before define the object should

    # Draw a rectangle around the body
    for (x, y, w, h) in nose:
        roi_next = next_image[y:y+h, x:x+w]   #(ycoord-start, ycoord-end)
        #roi_color = next_image[y:y+h, x:x+w]
        print(nose, "===============irung")
        print(roi_next, "===========roi")

    # save frame last detection
        cv2.imwrite("roi_next.png", roi_next)

    # labeled object while recognize
        color = (255, 0, 0)     #BGR 0-255
        stroke = 2
        weight = x + w      #end coord x
        height = y + h      #end coord y
        cv2.rectangle(frame, (x, y), (weight, height), color, stroke)

    # Optical Flow - Dense / Farneback
    flow = cv2.calcOpticalFlowFarneback(roi_temp, roi_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # show window frame
    cv2.imshow('face', frame)
    
    both = cv2.hconcat([roi_next, bgr])
    cv2.imshow('cam', both)

    
    # button
    k = cv2.waitKey(20) & 0xFF
    if k == ord('q'):                       # press q to quit
        break

cap.release()
cv2.destroyAllWindows()