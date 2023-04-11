from scipy.io import loadmat
import numpy as np
import cv2 as cv
from skimage.feature import hog
import os

# Function to process the annotations and extract relevant information (bounding boxes, ground truths)
def processAnnotations(directory):
    cars = []
    gts = []
    heights = []
    widths = []
    for filename in os.listdir(directory):
        # print(filename)
        image_id = filename.split('.')[0]
        label = loadmat('data/train_angle/labels/' + image_id + '.mat')['annotation'][0,0]
        image = cv.imread(directory + filename)
        class_values = label['class'].T
        bbox_values = label['bboxes']
        truncation_values = label['truncated'].T
        occlusion_values = label['occlusion'].T
        orientation_values = label['orient']
        for i in range(bbox_values.shape[0]):
            if class_values[i,0][0] != 'Car' or truncation_values[i,0][0] > 0.3 or occlusion_values[i,0][0] > 0.2:
                continue
            left, top, width, height = bbox_values[i]
            left, top, width, height = int(left), int(top), int(width), int(height)
            car = cv.cvtColor(image[top:top+height, left:left+width], cv.COLOR_BGR2GRAY)
            gt = round(orientation_values[i][0]) % 12
            cars.append(car)
            gts.append(gt)
            heights.append(height)
            widths.append(width)
    return cars, gts, heights, widths

# Function to extract features given bounding boxes
def extractFeatures(cars, heights, widths):
    avg_height = int(np.mean(heights))
    avg_width = int(np.mean(widths))
    # print(avg_height)
    # print(avg_width)
    features = []
    for i in range(len(cars)):
        cars[i] = cv.resize(cars[i], (avg_width, avg_height))
        feat_vec = hog(cars[i])
        features.append(feat_vec)
    return features

if __name__ == "__main__":
    directory = 'data/train_angle/image/'

    # Get the bounding boxes and ground truths of all the images
    cars, gts, heights, widths = processAnnotations(directory)

    # Extract features from the bounding boxes
    features = extractFeatures(cars, heights, widths)
    
    # Save the features and ground truths
    np.save('outputs/features.npy', features)
    np.save('outputs/gts.npy', gts)

    

