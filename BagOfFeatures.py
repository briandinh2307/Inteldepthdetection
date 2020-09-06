import cv2
import numpy as np

img = cv2.imread("Data/human.png", cv2.IMREAD_UNCHANGED)

orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(img, None)

img = cv2.drawKeypoints(img, keypoints, None)

cv2.imshow("Human", img)
cv2.waitKey(0)
cv2.destroyAllWindows()