"""
Get the image ready for processing

Boost contrast, greyscale etc.

Eoghan
"""
#edgedetection added onto eoghan's code

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

E = cv2.Canny(BINARY,threshold1=100,threshold2=200)

#code for probabilistic hough transform is from the following source https://docs.opencv.org/3.4.20/d9/db0/tutorial_hough_lines.html
linesP = cv2.HoughLinesP(E, 1, np.pi / 180, 50, None, 50, 10)

# Draw the lines
if linesP is not None:
    for line in linesP:
        l = line[0] if line.ndim == 2 else line
        cv2.line(BINARY_BGR, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (0, 0, 255), 3, cv2.LINE_AA)

# Show results using defined variables
cv2.imshow("Source Original", ORIGINAL)
cv2.imshow("Edges", E)
cv2.imshow("Detected Lines (Probabilistic Hough)", BINARY_BGR)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Save final processed image
cv2.imwrite('preprocessed_images/clock1_preprocessed2.png', BINARY_BGR)