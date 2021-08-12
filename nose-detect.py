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
roi_gray=[]

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = rescale_frame(frame, percent=rescale_percent)
    
    # Detect ROI
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    nose = noseCascade.detectMultiScale(frame, scaleFactor=5, minNeighbors=1)
    #scaleFactor value define range between camera and the object
    #minNeighbors specifying how many neighbors each candidate rectangle should have 
        #before define the object should detected
    print(nose)

    # Draw a rectangle around the object
    for (x, y, w, h) in nose:
        roi_gray = frame[y:y+h, x:x+w]   #(ycoord-start, ycoord-end)
        #roi_color = next_image[y:y+h, x:x+w]

    # save frame last detection
        cv2.imwrite("image1.png", roi_gray) #(img_item, roi)
        cv2.imwrite("image2.png", frame)
        np.savetxt("test1.txt", roi_gray, fmt='%.5e', delimiter=',')

    # labeled object while recognize
        color = (255, 0, 0)     #BGR 0-255
        stroke = 2
        weight = x + w      #end coord x
        height = y + h      #end coord y
        cv2.rectangle(roi_gray, (x, y), (weight, height), color, stroke)

    # show window frame
    #both = cv2.hconcat([frame, vid_roi])
    cv2.imshow('cam', frame)
    cv2.imshow('roi', roi_gray)
    
    # button
    k = cv2.waitKey(20) & 0xFF
    if k == ord('q'):                       # press q to quit
        break
    elif k == ord('s'):                     # press s to write image
        cv2.imwrite('frame.png', frame)

cap.release()
cv2.destroyAllWindows()