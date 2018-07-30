import matplotlib.pyplot as plt
import numpy as np
import cv2
import lane_detection3

image = cv2.imread('/Users/edwardkim/Downloads/webotsEx-master/controllers/foo/webots_pics/yo70.png')
# cv2.imshow("actual", image)
# cv2.imshow("cropped", image[0:200, 0:100])
# cv2.waitKey(0)

img = cv2.resize(image, (960, 540))
gray_image = lane_detection3.discard_colors(img)
image = lane_detection3.detect_edges(gray_image, low_threshold=50, high_threshold=150)

# plt.imshow(image)
# plt.pause(5)

# xsize = image.shape[1]
# ysize = image.shape[0]
# print(xsize, ysize)
# dx1 = int(0 * xsize)
# dx2 = int(0.4 * xsize)
# dy = int(0.3 * ysize)

# calculate vertices for region of interest
vertices = np.array([[(0, 540), (350, 300), (600, 300), (800, 540)]], dtype=np.int32)
# print(vertices)

image = lane_detection3.region_of_interest(image, vertices)

# plt.imshow(image)
# plt.pause(10)

rho = 0.8
theta = np.pi/180
threshold = 25
min_line_len = 50
max_line_gap = 200

lines = lane_detection3.hough_lines(image, rho, theta, threshold, min_line_len, max_line_gap)
# print("lines : ")
# print(lines)
right_lines, left_lines = lane_detection3.separate_lines(lines)
# print(right_lines)
# print(left_lines)

if right_lines and left_lines:
    right = lane_detection3.reject_outliers(right_lines,  cutoff=(0.75, 0.95))
    left = lane_detection3.reject_outliers(left_lines, cutoff=(-0.65, -0.45))

# print("right and left:")
# print(right)
# print(left)

######### Run regression on points found for the right lane #########

xr, yr, mr, cr = lane_detection3.lines_linreg(right)
# This variable represents the top-most point in the image where we can reasonable draw a line to.
min_yr = np.min(yr)
# Calculate the top point using the slopes and intercepts we got from linear regression.
top_point = np.array([(min_yr - cr) / mr, min_yr], dtype=int)

# Repeat this process to find the bottom right point.
max_y = np.max(yr)
bot_point = np.array([(max_y - cr) / mr, max_y], dtype=int)

x1r, y1r = lane_detection3.extend_point(bot_point[0],bot_point[1],top_point[0],top_point[1], -1000) # bottom point
x2r, y2r = lane_detection3.extend_point(bot_point[0],bot_point[1],top_point[0],top_point[1],  1000) # top point

# return the line.
right_line = np.array([[x1r,y1r,x2r,y2r]]) ####FIX  SOMETHING HERE1!!!!!!!



######### Run regression on points found for the left lane #########
xl, yl, ml, cl = lane_detection3.lines_linreg(left)
# This variable represents the top-most point in the image where we can reasonable draw a line to.
min_y = np.min(yl)
# Calculate the top point using the slopes and intercepts we got from linear regression.
top_point = np.array([(min_y - cl) / ml, min_y], dtype=int)

# Repeat this process to find the bottom left point.
max_y = np.max(yl)
bot_point = np.array([(max_y - cl) / ml, max_y], dtype=int)

x1e, y1e = lane_detection3.extend_point(bot_point[0],bot_point[1],top_point[0],top_point[1], -1000) # bottom point
x2e, y2e = lane_detection3.extend_point(bot_point[0],bot_point[1],top_point[0],top_point[1],  1000) # top point

left_line = np.array([[x1e,y1e,x2e,y2e]])

## compute the points that define middle lane by finding avg of left/right lane
mid_line = (left_line+right_line)/2 # convert the float array to int array

# convert the format of the lines into drawable form
mid_line = np.array([mid_line], dtype=np.int32)
right_line = np.array([right_line], dtype=np.int32)
left_line = np.array([left_line], dtype=np.int32)

# append all line information to lines
lines = np.append(right_line, left_line, axis=0)
lines = np.append(lines, mid_line, axis = 0)

line_image = np.copy(img*0)
lane_detection3.draw_lines(line_image, lines, thickness=10)
# plt.imshow(line_image)
# plt.pause(5)
# lane_detection3.draw_lines(line_image, np.array([left_line], dtype=np.int32), thickness=3)

line_image = lane_detection3.region_of_interest(line_image, vertices)
# plt.imshow(line_image)
# plt.pause(5)

final_image = lane_detection3.weighted_image(line_image, img)

#### compute intersecting midpoint
yp = [300, 350, 400, 450, 500] ## set vertical point of interest
for y in yp:
	xleft = (y-cl)/ml
	xright = (y-cr)/mr
	x_mid = int((xleft+xright)/2)
	cv2.circle(final_image ,(int(x_mid),int(y)),10,(0,255,0),-11)

plt.imshow(final_image)
plt.pause(50)

