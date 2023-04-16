import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from depth import computeDisparity, computeDepth, getCameraParams
from skimage.measure import regionprops
from plane import convert2DTo3D

# COMPUTE FEATURES OF IMAGES
# Features include:
#   - Centroid of segment (x,y)
#   - RGB of centroid of segment (R,G,B)
#   - 3D representation of centroid of segment (X,Y,Z)

def getFeatures(mode):
    directory = f'data/{mode}/image_left'
    features = []
    prefix_dict = {'um': 0, 'umm': 1, 'uu': 2}

    for filename in os.listdir(directory):
        # print(filename)
        prefix = filename.split('.')[0].split('_')[0]  
        image_id = filename.split('.')[0].split('_')[1]

        id_1 = prefix_dict[prefix]
        id_2 = int(image_id)

        left_image_file = f'data/{mode}/image_left/{filename}'
        right_image_file = f'data/{mode}/image_right/{filename}'
        calib_file = f'data/{mode}/calib/{prefix}_{image_id}.txt'

        left_image = cv.imread(left_image_file)
        right_image = cv.imread(right_image_file)

        if mode == 'train':
            left_image_gt_file = f'data/{mode}/gt_image_left/{prefix}_road_{image_id}.png'
            left_image_gt = cv.imread(left_image_gt_file)

        gray_left_image = cv.cvtColor(left_image, cv.COLOR_BGR2GRAY)
        gray_right_image = cv.cvtColor(right_image, cv.COLOR_BGR2GRAY)

        f, T, px, py = getCameraParams(calib_file)
        disparity = computeDisparity(gray_left_image, gray_right_image, 64, 9, 1)
        depth = computeDepth(disparity, T, f)

        segments = slic(left_image, n_segments=1000, sigma=5)
        regions = regionprops(segments)

        # plt.imshow(mark_boundaries(left_image, segments))
        # plt.show()

        for region in regions:
            id_3 = region.label
            cx, cy = region.centroid
            cz = depth[int(cx), int(cy)]
            cX, cY, cZ = convert2DTo3D(cx, cy, cz, px, py, f)
            B, G, R = left_image[int(cx), int(cy)]

            if mode == 'train':
                gt_centre = left_image_gt[int(cx), int(cy)]
                label = 0
                if gt_centre[0] == 255:
                    label = 1
                feat_vec = np.array([id_1, id_2, id_3, cx, cy, R, G, B, cX, cY, cZ, label])
            else:
                feat_vec = np.array([id_1, id_2, id_3, cx, cy, R, G, B, cX, cY, cZ])
            
            features.append(feat_vec)
    
    return features

if __name__ == "__main__":
    train_features = getFeatures('train')
    test_features = getFeatures('test')

    np.savetxt('outputs/train_features.csv', train_features, delimiter=',')
    np.savetxt('outputs/test_features.csv', test_features, delimiter=',')