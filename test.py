import numpy as np
import cv2

# frame1 = np.loadtxt("test1.txt")
# prev_image = frame1
# hsv = np.zeros_like(frame1)
# hsv[..., 1] = 255

# print(frame1)
# np.savetxt("test2.txt", frame1, fmt='%.5e', delimiter=',')

# #cv2.cvtColor(frame1, cv2.COLOR_GRAY2RGB)
# cv2.imshow('test', frame1)
# cv2.imshow('test2', hsv)
# cv2.waitKey(0) # waits until a key is pressed
# cv2.destroyAllWindows() # destroys the window showing image

frame1 = np.loadtxt("test1.txt", delimiter=',', dtype='float')
prev_image = frame1
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255
print(hsv)
np.savetxt("test-hsv.txt", hsv, fmt='%.5e', delimiter=',')