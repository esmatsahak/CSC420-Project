import cv2 as cv
import numpy as np
import json
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report    
from skimage.segmentation import slic

# Function to train classifier and predict test labels
def classifier(clf, train_input, train_labels, test_input):
    clf.fit(train_input, train_labels)
    train_output = clf.predict(train_input)
    print(classification_report(train_labels, train_output))
    test_output = clf.predict(test_input)
    return test_output

# Function to identify road pixels per test image
def getRoadDict(road_ids):
    prefix_dict = {0: 'um', 1: 'umm', 2: 'uu'}
    road_dict = {}
    for road_id in road_ids:
        prefix = prefix_dict[int(road_id[0])]
        image_id = int(road_id[1])
        if image_id < 10:
            id = f'00000{image_id}'
        else:
            id = f'0000{image_id}'
        key = f'{prefix}_{id}'    
        if road_dict.get(key, 0) == 0:
            road_dict[key] = [road_id[2]]
        else:
            road_dict[key].append(road_id[2])
    return road_dict

# Function to mark road pixels in test images
def markRoadPixels(directory, road_dict):
    for key in road_dict:
        filename = f'{directory}{key}.jpg'
        # print(filename)
        image = cv.imread(filename)
        segments = slic(image, n_segments=1000, sigma=5)
        for i in road_dict[key]:
            road_idx = np.where(segments == i)
            image[road_idx] = [0, 255, 0]
        cv.imwrite(f'outputs/road/{key}.jpg', image)

if __name__ == '__main__':
    # Read train and test features
    train_features = np.genfromtxt('outputs/train_features.csv', delimiter=',')
    test_features = np.genfromtxt('outputs/test_features.csv', delimiter=',')

    # Split features into ids, input and labels
    train_ids = train_features[:, 0:3]
    train_input = train_features[:, 3:11]
    train_labels = train_features[:, -1]
    test_ids = test_features[:, 0:3]
    test_input = test_features[:, 3:]

    # Train MLP classifier and predict test labels
    mlp = MLPClassifier()
    test_labels = classifier(mlp, train_input, train_labels, test_input)

    # List road pixels per test image in a dictionary
    idx = np.nonzero(test_labels)
    road_ids = test_ids[idx]
    road_dict = getRoadDict(road_ids)

    # Save road dictionary to json file
    with open('outputs/road_dict.json', 'w') as fp:
        json.dump(road_dict, fp)
    
    # Mark road pixels in test images
    directory = 'data/test/image_left/'
    markRoadPixels(directory, road_dict)