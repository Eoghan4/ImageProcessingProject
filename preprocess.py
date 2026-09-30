"""
Get the image ready for processing

Boost contrast, greyscale etc.

Eoghan
"""

import easygui
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as image

I = cv2.imread("source_images/clock1.png")

ORIGINAL = I.copy()

HEIGHT, WIDTH, _ = I.shape

# Convert to Greyscale
GRAY = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

# Boost Brightness
BRIGHT = cv2.convertScaleAbs(GRAY, alpha=1, beta=100)

# Make Binary
BINARY = cv2.threshold(BRIGHT,180,255,cv2.THRESH_BINARY)[1]

# Convert back to BGR to add a coloured indicator for middle of hands
BINARY_BGR = cv2.cvtColor(BINARY, cv2.COLOR_GRAY2BGR)
cv2.circle(img = BINARY_BGR, center = (WIDTH//2,HEIGHT//2), radius = 2, color = (255,0,255), thickness = 5)

cv2.imshow("Boost", BINARY_BGR)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('preprocessed_images/clock1_preprocessed.png', BINARY_BGR)