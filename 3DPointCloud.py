import cv2 as cv
import numpy as np
import open3d as o3d
import json
from depth import computeDisparity, computeDepth, getCameraParams
from plane import convert2DTo3D, getRoadPixels, preprocessRoadPixels, getBestPlane

# Function to generate point cloud of image and road plane
def convertToPointCloud(image, depth, plane, f, px, py):    
    point_cloud = o3d.geometry.PointCloud()
    plane_cloud = o3d.geometry.PointCloud()
    h, w = image.shape[:2]
    points = []
    plane_points = []
    for x in range(h):
        for y in range(w):
            z = depth[x,y]
            X, Y, Z = convert2DTo3D(x, y, z, px, py, f)
            Z_plane = plane[0]*X + plane[1]*Y + plane[2]
            points += [[X, Y, Z]]
            plane_points += [[X, Y, -Z_plane]]
    point_cloud.points = o3d.utility.Vector3dVector(points)
    plane_cloud.points = o3d.utility.Vector3dVector(plane_points)
    point_cloud.colors = o3d.utility.Vector3dVector(image.reshape(-1,3)/255.0)
    plane_cloud.colors = o3d.utility.Vector3dVector([[1,0,0] for i in range(h*w)])
    return point_cloud, plane_cloud

# Write code to use open3D TO visualize a point cloud of the file data/train/image_left/um_000000.jpg
if __name__ == "__main__":
    # Load the image
    image_id = "umm_000085"
    left_image = cv.imread(f"data/test/image_left/{image_id}.jpg")
    right_image = cv.imread(f"data/test/image_right/{image_id}.jpg")

    gray_left_image = cv.cvtColor(left_image, cv.COLOR_BGR2GRAY)
    gray_right_image = cv.cvtColor(right_image, cv.COLOR_BGR2GRAY)

    # Get the camera parameters
    f, T, px, py = getCameraParams(f"data/test/calib/{image_id}.txt")

    # Compute the depth  map
    disparity = computeDisparity(gray_left_image, gray_right_image, 128, 5, 4)
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

    # Convert the image to a point cloud
    rgb_image = cv.cvtColor(left_image, cv.COLOR_BGR2RGB)
    point_cloud, plane_cloud = convertToPointCloud(rgb_image, depth, plane, f, px, py)
    
    # Visualize the point cloud
    o3d.visualization.draw_geometries([plane_cloud, point_cloud])