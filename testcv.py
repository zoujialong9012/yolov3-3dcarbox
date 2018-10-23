import cv2
import numpy as np

print("first window 1")

cv2.namedWindow('test', cv2.WINDOW_AUTOSIZE)
black = np.zeros((400, 400, 3), np.uint8)
cv2.imshow('test', black)
cv2.waitKey(0)

print("second window 2")

cv2.imshow('test', black)
cv2.waitKey(0)

print("third window 3")

cv2.imshow('test', black)
cv2.waitKey(0)

cv2.destroyAllWindows()
