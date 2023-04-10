import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from depth import computeDisparity, computeDepth, getCameraParams

# COMPUTE FEATURES OF IMAGES
# Features include:
#   - Normalized centre (x,y) of segment
#   - Mean colour (RGB) of segment
#   - Mean depth (z) of segment

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

        f, T = getCameraParams(calib_file)
        disparity = computeDisparity(gray_left_image, gray_right_image, 64, 9, 1)
        depth = computeDepth(disparity, T, f)

        segments = slic(left_image, n_segments=1000, sigma=5)

        # plt.imshow(mark_boundaries(left_image, segments))
        # plt.show()

        for i in range(0, np.amax(segments)+1):
            id_3 = i
            idx = np.where(segments == i)
            depth_idx = cv.normalize(depth[idx], None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
            left_image_idx = left_image[idx]/255
            idx_norm = [idx[0]/left_image.shape[0], idx[1]/left_image.shape[1]]

            centre_norm = np.mean(idx_norm, axis=1)
            mean_color = np.mean(left_image_idx, axis=0)
            mean_depth = np.mean(depth_idx)

            if mode == 'train':
                centre = np.mean(idx, axis=1)
                gt_centre = left_image_gt[int(centre[0]), int(centre[1])]
                label = 0
                if gt_centre[0] == 255:
                    label = 1
                feat_vec = np.array([id_1, id_2, id_3, centre_norm[0], centre_norm[1], mean_color[0], mean_color[1], mean_color[2], mean_depth, label])
            else:
                feat_vec = np.array([id_1, id_2, id_3, centre_norm[0], centre_norm[1], mean_color[0], mean_color[1], mean_color[2], mean_depth])
            
            features.append(feat_vec)
    
    return features

if __name__ == "__main__":
    train_features = getFeatures('train')
    # print()
    test_features = getFeatures('test')

    np.savetxt('outputs/train_features.csv', train_features, delimiter=',')
    np.savetxt('outputs/test_features.csv', test_features, delimiter=',')