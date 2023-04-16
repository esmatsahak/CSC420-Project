import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Compute disparity of images
def computeDisparity(image_left, image_right, num_disparities, block_size, min_disparity):
    stereo = cv.StereoSGBM_create(numDisparities=num_disparities, blockSize=block_size, minDisparity=min_disparity)
    disparity = stereo.compute(image_left, image_right).astype(np.float32)/16
    return disparity

# Compute depth of images
def computeDepth(disparity_map, T, f):
    depth = np.divide(f * T, disparity_map, where=disparity_map!=0)
    return depth

# Get camera parameters from calibration file
def getCameraParams(calib_file):
    with open(calib_file) as f:
        line = f.readlines()[1].split()
        f = float(line[1])
        T = -float(line[4])/f 
        px = float(line[3])
        py = float(line[7])
    return f, T, px, py

if __name__ == "__main__":
    # Image ID
    image_id = 'umm_000000'

    # Camera parameters (from calibration file)
    calib_file = f'data/train/calib/{image_id}.txt'
    f, T, px, py = getCameraParams(calib_file)

    # Read images
    image_left = cv.imread(f'data/train/image_left/{image_id}.jpg', cv.IMREAD_GRAYSCALE)
    image_right = cv.imread(f'data/train/image_right/{image_id}.jpg', cv.IMREAD_GRAYSCALE)

    # Compute disparity and visualize
    disparity = computeDisparity(image_left, image_right, 64, 9, 1)
    disparity_map = cv.normalize(disparity, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    cv.imwrite(f'outputs/disparity_map_{image_id}.jpg', disparity_map)

    # Compute depth and visualize
    depth = computeDepth(disparity, T, f)
    depth_map = cv.normalize(depth, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    cv.imwrite(f'outputs/depth_map_{image_id}.jpg', depth)