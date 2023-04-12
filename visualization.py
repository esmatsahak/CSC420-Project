from objectDetection import get_prediction
from viewpointFeatures import extractFeatures
import torchvision
import cv2 as cv
import numpy as np
import pickle
import os

if __name__ == '__main__':
    # Load object detection model
    COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # Load viewpoint classifier
    directory = "data/test/image_left/"
    classifier = pickle.load(open("outputs/model.pkl", "rb"))

    # Iterate through test images
    for filename in os.listdir(directory):
        print(filename)

        # Load image
        img_path = directory + filename
        image = cv.imread(img_path)

        # Get car bounding boxes of image
        boxes, pred_cls = get_prediction(img_path, 0.5, model, COCO_INSTANCE_CATEGORY_NAMES)
        car_idx = np.where(np.array(pred_cls) == 'car')
        car_boxes = np.array(boxes)[car_idx]
        cars = []
        centres = []
        for box in car_boxes:
            x0 = int(box[0][0])
            y0 = int(box[0][1])
            x1 = int(box[1][0])
            y1 = int(box[1][1])
            xc = int((x0 + x1)/2)
            yc = int((y0 + y1)/2)
            centres.append((xc, yc))
            car = image[y0:y1, x0:x1]
            cars.append(car)
            cv.rectangle(image, (x0, y0), (x1, y1), color=(0,255,0), thickness=2) 

        if len(cars) != 0:
            # Predict viewpoint of cars in image
            features = extractFeatures(cars, 60, 115)
            predictions = classifier.predict(features)
            angles = [30 * pred for pred in predictions]

            # Draw arrow of car viewpoints
            for i, angle in enumerate(angles):
                xs = centres[i][0]
                ys = centres[i][1]
                xe = xs + int(50 * np.cos(np.deg2rad(angle+90)))
                ye = ys + int(50 * np.sin(np.deg2rad(angle+90)))
                cv.arrowedLine(image, (xs, ys), (xe, ye), color=(0,0,255), thickness=1)
                cv.putText(image, str(angle), (xs, ys), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1, cv.LINE_AA)

        # Save image
        cv.imwrite("outputs/viewpoints/" + filename, image)