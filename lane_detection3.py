# Source : https://medium.com/@ldesegur/a-lane-detection-approach-for-self-driving-vehicles-c5ae1679f7ee
import matplotlib.pyplot as plt
import numpy as np
import cv2

def remove_noise(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def discard_colors(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def detect_edges(image, low_threshold, high_threshold):
    return cv2.Canny(image, low_threshold, high_threshold)

# image = detect_edges(gray_image, low_threshold=50, high_threshold=150)
# plt.imshow(image, cmap='gray')

def region_of_interest(image, vertices):

    # defining a blank mask to start with mask = np.zeros_like(image)
    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if (len(image.shape)>2):
    	channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image ignore_mask_color = (255,) * channel_count
    	ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    mask = np.zeros_like(image)
    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)
  
  # returning the image only where mask pixels are non-zero
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# xsize = img.shape[1]
# ysize = img.shape[0]
# dx1 = int(0.0725 * xsize)
# dx2 = int(0.425 * xsize)
# dy = int(0.6 * ysize)
# # calculate vertices for region of interest
# vertices = np.array([[(dx1, ysize), (dx2, dy), (xsize - dx2, dy), (xsize - dx1, ysize)]], dtype=np.int32)
# image = region_of_interest(image, vertices)

def draw_lines(image, lines, color=[255, 0, 0], thickness=5):

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)


def hough_lines(image, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

# rho = 0.8
# theta = np.pi/180
# threshold = 25
# min_line_len = 50
# max_line_gap = 200
# lines = hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap)

def slope(x1, y1, x2, y2):
    return (y1 - y2) / (x1 - x2)

def separate_lines(lines):
    right = []
    left = []
    for x1,y1,x2,y2 in lines[:, 0]:
        m = slope(x1,y1,x2,y2)
        if m >= 0:
            right.append([x1,y1,x2,y2,m])
        else:
            left.append([x1,y1,x2,y2,m])
    return right, left

# right_lines, left_lines = separate_lines(lines)

def reject_outliers(data, cutoff, threshold=0.08):
    data = np.array(data)
    data = data[(data[:, 4] >= cutoff[0]) & (data[:, 4] <= cutoff[1])]
    m = np.mean(data[:, 4], axis=0)
    return data[(data[:, 4] <= m+threshold) & (data[:, 4] >= m-threshold)]

# if right_lines and left_lines:
#     right = reject_outliers(right_lines,  cutoff=(0.45, 0.75))
#     left = reject_outliers(left_lines, cutoff=(-0.85, -0.6))

def lines_linreg(lines_array):

    x = np.reshape(lines_array[:, [0, 2]], (1, len(lines_array) * 2))[0]
    y = np.reshape(lines_array[:, [1, 3]], (1, len(lines_array) * 2))[0]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    x = np.array(x)
    y = np.array(x * m + c)
    return x, y, m, c

# x, y, m, c = lines_linreg(lines)
# # This variable represents the top-most point in the image where we can reasonable draw a line to.
# min_y = np.min(y)
# # Calculate the top point using the slopes and intercepts we got from linear regression.
# top_point = np.array([(min_y - c) / m, min_y], dtype=int)
# # Repeat this process to find the bottom left point.
# max_y = np.max(y)
# bot_point = np.array([(max_y - c) / m, max_y], dtype=int)

def extend_point(x1, y1, x2, y2, length):
    line_len = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    x = x2 + (x2 - x1) / line_len * length
    y = y2 + (y2 - y1) / line_len * length
    return x, y

# x1e, y1e = extend_point(bot_point[0],bot_point[1],top_point[0],top_point[1], -1000) # bottom point
# x2e, y2e = extend_point(bot_point[0],bot_point[1],top_point[0],top_point[1],  1000) # top point
# # return the line.
# line = np.array([[x1e,y1e,x2e,y2e]])
# return np.array([line], dtype=np.int32)


def weighted_image(image, initial_image, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_image, α, image, β, λ)

# line_image = np.copy((image)*0)
# draw_lines(line_image, lines, thickness=3)
# line_image = region_of_interest(line_image, vertices)
# final_image = weighted_image(line_image, image)
