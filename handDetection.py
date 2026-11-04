"""
Use Jacks Edge Detection to isolate the clock hands

Get clean output ready for angle calculation

Eoghan
"""

import math
import cv2
import numpy as np

CENTRE_PASS_THRESHOLD = 20
ANGLE_CLUSTER_THRESHOLD = 0.25


def line_distance_to_point(x1, y1, x2, y2, px, py):
    num = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
    den = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    return num / den if den > 0 else float('inf')


def line_length(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def endpoint_furthest_from_centre(x1, y1, x2, y2, cx, cy):
    d1 = math.sqrt((x1 - cx) ** 2 + (y1 - cy) ** 2)
    d2 = math.sqrt((x2 - cx) ** 2 + (y2 - cy) ** 2)
    return (x1, y1) if d1 > d2 else (x2, y2)


def cluster_lines(lines, cx, cy):
    buckets = []
    for line in lines:
        x1, y1, x2, y2 = line
        tip = endpoint_furthest_from_centre(x1, y1, x2, y2, cx, cy)
        angle = math.atan2(tip[1] - cy, tip[0] - cx)
        placed = False
        for bucket in buckets:
            if abs(angle - bucket['angle']) < ANGLE_CLUSTER_THRESHOLD:
                bucket['lines'].append(line)
                bucket['angle'] = (bucket['angle'] + angle) / 2
                placed = True
                break
        if not placed:
            buckets.append({'angle': angle, 'lines': [line]})
    return buckets


def best_line_for_cluster(cluster, cx, cy):
    return max(cluster['lines'], key=lambda l: line_length(*l))


img = cv2.imread("preprocessed_images/clock1_preprocessed.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

HEIGHT, WIDTH = gray.shape
cx, cy = WIDTH // 2, HEIGHT // 2

edges = cv2.Canny(gray, 100, 200)

#code for probabilistic hough transform is from the following source https://docs.opencv.org/3.4.20/d9/db0/tutorial_hough_lines.html
raw_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 30, 10)

if raw_lines is None:
    print("No lines detected.")
    exit()

centre_lines = []
for line in raw_lines:
    l = line[0] if line.ndim == 2 else line
    x1, y1, x2, y2 = map(int, l)
    dist = line_distance_to_point(x1, y1, x2, y2, cx, cy)
    if dist < CENTRE_PASS_THRESHOLD:
        centre_lines.append((x1, y1, x2, y2))

print(f"Lines passing through centre: {len(centre_lines)}")

clusters = cluster_lines(centre_lines, cx, cy)
clusters.sort(key=lambda c: len(c['lines']), reverse=True)
top2 = clusters[:2]

if len(top2) < 2:
    print("Could not find two distinct hands.")
    exit()

hand_a = best_line_for_cluster(top2[0], cx, cy)
hand_b = best_line_for_cluster(top2[1], cx, cy)

tip_a = endpoint_furthest_from_centre(*hand_a, cx, cy)
tip_b = endpoint_furthest_from_centre(*hand_b, cx, cy)

len_a = line_length(*hand_a)
len_b = line_length(*hand_b)

if len_a >= len_b:
    minute_hand, minute_tip = hand_a, tip_a
    hour_hand,   hour_tip   = hand_b, tip_b
else:
    minute_hand, minute_tip = hand_b, tip_b
    hour_hand,   hour_tip   = hand_a, tip_a

print(f"Minute hand tip: {minute_tip}")
print(f"Hour hand tip:   {hour_tip}")

out = img.copy()

cv2.line(out, (cx, cy), minute_tip, (255, 0, 0), 3, cv2.LINE_AA)
cv2.circle(out, minute_tip, 5, (255, 0, 0), -1)

cv2.line(out, (cx, cy), hour_tip, (0, 200, 0), 3, cv2.LINE_AA)
cv2.circle(out, hour_tip, 5, (0, 200, 0), -1)

cv2.circle(out, (cx, cy), 5, (0, 0, 255), -1)

cv2.imshow("Hand Detection", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("preprocessed_images/clock1_hands.png", out)
