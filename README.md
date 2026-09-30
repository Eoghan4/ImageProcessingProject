# Image Processing Project - Time Teller
#### Read an image of an analogue clock, and output the time.

## Task 1: Pre-processing (Eoghan)
Get the image in the best possible state in order to extract the location of the hands. Involves, boosting contrast, greyscale, removing unnecassary details.

Currently, **preprocess.py** takes in **source_images/clock1.png** and turns it into a binary image, in order to increase the contrast between the hands and the rest of the clock. It then converts it back to BGR, and adds a magenta circle in order to make it easier to find the centre of the clock.

## Task 2: Hand Detection (Jack)
Locate where the clock hands are in the pre-processed image, then determine the co-ordinates of the start and end points as well as which hand is the hour hand and which is the minute hand to provide to Daniel

## Task 3: Time Calculation (Daniel)
Task Description
