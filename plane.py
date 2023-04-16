import cv2 as cv
import numpy as np
import json
import matplotlib.pyplot as plt
import os
from depth import computeDisparity, computeDepth, getCameraParams
from skimage.segmentation import slic
from skimage.measure import regionprops
from sklearn.linear_model import RANSACRegressor

# Function to get world coordinates of image pixel 
# OpenCV swaps X and Y so we swap them back, in 3D Z is y-axis and Y is z-axis
def convert2DTo3D(x, y, z, f, px, py):
    Z = z
    X = (x-px)*Z/f
    Y = (y-py)*Z/f
    return Y, Z, X

# Function to get centroids of image superpixels previously identified to be part of the road
def getRoadPixels(image, road_dict, image_id):
    segments = slic(image, n_segments=1000, sigma=5)
    regions = regionprops(segments)
    road_pixels = [[], []]
    for i in road_dict[image_id]:
        region = regions[int(i-1)]
        cx, cy = region.centroid
        road_pixels[0].append(int(cx))
        road_pixels[1].append(int(cy))
    return road_pixels

# Function to convert road pixels to 3D points
def preprocessRoadPixels(road_pixels, depth, f, px, py):
    road_z = depth[road_pixels[0], road_pixels[1]]
    road_pixels += [road_z]
    road_points = np.zeros((len(road_pixels[0]), 3))
    for i in range(len(road_pixels[0])):
        x = road_pixels[0][i]
        y = road_pixels[1][i]
        z = road_pixels[2][i]
        X, Y, Z = convert2DTo3D(x, y, z, f, px, py) # World-coordinates
        road_points[i] = [X, Y, Z] 
    return road_points

# Function to get the best plane (using RANSAC) that fits the sample road points
def getBestPlane(road_points):
    xy = road_points[:, :2]
    z = road_points[:, 2]
    ransac = RANSACRegressor()
    ransac.fit(xy, z)
    inlier_count = np.sum(ransac.inlier_mask_)
    plane = np.append(ransac.estimator_.coef_, ransac.estimator_.intercept_)
    return plane, inlier_count

# Function to plot road points and road plane
def plot3DPoints(points, plane):
    ax = plt.axes(projection='3d')
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    m_x, m_y, b = plane[0], plane[1], plane[2]
    ax.scatter(x, y, z)
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    pt_x = np.linspace(min_x, max_x, 100)
    pt_y = np.linspace(min_y, max_y, 100)
    pt_X, pt_Y = np.meshgrid(pt_x, pt_y)
    pt_Z = m_x*pt_X + m_y*pt_Y + b
    ax.plot_surface(pt_X, pt_Y, pt_Z, alpha=0.2)
    return ax

if __name__ == "__main__":
    directory = "data/test/image_left/"

    # Iterate through all test images
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            # Load test image
            image_id = filename.split(".")[0]
            # print(image_id)
            left_image = cv.imread(f"data/test/image_left/{image_id}.jpg")
            right_image = cv.imread(f"data/test/image_right/{image_id}.jpg")

            # Compute depth of image pixels
            f, T, px, py = getCameraParams(f"data/test/calib/{image_id}.txt")
            disparity = computeDisparity(left_image, right_image, 64, 9, 1)
            depth = computeDepth(disparity, T, f)

            # Get sample road points of image
            with open("outputs/road_dict.json", "r") as file:
                road_dict = json.load(file)
            road_pixels = getRoadPixels(left_image, road_dict, image_id)
            road_points = preprocessRoadPixels(road_pixels, depth, f, px, py)
            
            # Get best plane that fits the road points
            plane, inlier_count = getBestPlane(road_points)
            # print(inlier_count)
            # print(plane)

            # Plot 3D points and plane
            ax = plot3DPoints(road_points, plane)
            plt.savefig(f"outputs/planes/{image_id}.jpg")
            plt.close()