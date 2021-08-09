import sys
import cv2

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

# function for change resolution
def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)

# Create the haar cascade
noseCascade = cv2.CascadeClassifier('haarcascade_mcs_nose')

# Capture frame as video stream
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = rescale_frame(frame, percent=75)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = noseCascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]   #(ycoord-start, ycoord-end)
        #roi_color = frame[y:y+h, x:x+w]

    # save frame last detection
        img_item = "image.png"
        cv2.imwrite(img_item, frame) #(img_item, roi_gray)

    # labeled object while recognize
        color = (255, 0, 0)     #BGR 0-255
        stroke = 2
        weight = x + w      #end coord x
        height = y + h      #end coord y
        cv2.rectangle(frame, (x, y), (weight, height), color, stroke)

    # show window frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()