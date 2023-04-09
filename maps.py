import cv2 as cv
import numpy as np

# Compute disparity map
# Source: https://docs.opencv.org/3.4/dd/d53/tutorial_py_depthmap.html
def computeDisparityMap(image_left, image_right, num_disparities, block_size):
    stereo = cv.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
    disparity_map = stereo.compute(image_left, image_right)
    return disparity_map

# Compute depth map
def computeDepthMap(disparity_map, T, f):
    depth_map = np.divide(f * T, disparity_map, where=disparity_map!=0)
    cv.normalize(depth_map, depth_map, 0, 255, cv.NORM_MINMAX)
    return depth_map

if __name__ == "__main__":
    image_id = 'umm_000000'

    # Camera parameters (from calibration file)
    T = 0.54
    f = 721.5

    # Read images
    image_left = cv.imread(f'data/train/image_left/{image_id}.jpg', cv.IMREAD_GRAYSCALE)
    image_right = cv.imread(f'data/train/image_right/{image_id}.jpg', cv.IMREAD_GRAYSCALE)

    # Compute disparity map
    disparity_map = computeDisparityMap(image_left, image_right, 128, 5)
    cv.imwrite(f'outputs/disparity_map_{image_id}.jpg', disparity_map)

    # Compute depth map
    depth_map = computeDepthMap(disparity_map, T, f)
    cv.imwrite(f'outputs/depth_map_{image_id}.jpg', depth_map)
