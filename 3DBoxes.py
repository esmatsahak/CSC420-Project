import torchvision
import cv2 as cv
import numpy as np
import json
import matplotlib.pyplot as plt
import os
from objectDetection import get_prediction
from depth import computeDisparity, computeDepth, getCameraParams
from plane import getRoadPixels, preprocessRoadPixels, getBestPlane, plot3DPoints

if __name__ == '__main__':
    # Load object detection model
    COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # Iterate over test images
    directory = "data/test/image_left/"
    for filename in os.listdir(directory):
        # Load image
        image_id = filename.split(".")[0]
        # print(image_id)
        img_path = f"data/test/image_left/{image_id}.jpg"
        left_image = cv.imread(img_path)
        right_image = cv.imread(f"data/test/image_right/{image_id}.jpg")

        # Get depth map of image
        f, T = getCameraParams(f"data/test/calib/{image_id}.txt")
        disparity = computeDisparity(left_image, right_image, 64, 9, 1)
        depth = computeDepth(disparity, T, f)

        # Get road plane
        with open("outputs/road_dict.json", "r") as f:
            road_dict = json.load(f)
        road_pixels = getRoadPixels(left_image, road_dict, image_id)
        road_points = preprocessRoadPixels(road_pixels, depth)  
        plane, _ = getBestPlane(road_points)  

        # Get bounding boxes of cars in image
        boxes, pred_cls = get_prediction(img_path, 0.5, model, COCO_INSTANCE_CATEGORY_NAMES)
        car_idx = np.where(np.array(pred_cls) == 'car')
        car_boxes = np.array(boxes)[car_idx]

        # Plot road points and road plane
        ax = plot3DPoints(road_points, plane, left_image)

        # Iterate over bounding boxes
        for box in car_boxes:
            # Get top-left and bottom-right corners of bounding box
            y0 = int(box[0][0])   
            x0 = int(box[0][1])   
            y1 = int(box[1][0])   
            x1 = int(box[1][1])   
            # Get center of bounding box
            xc = int((x0+x1)/2)
            yc = int((y0+y1)/2)
            # Get depth of center point
            zc = depth[xc, yc]
            # Get depth of point on road plane directly below center point
            zr = plane[2] - plane[0]*xc - plane[1]*yc
            # Get distance between center point and each face of 3D bounding box
            dx = abs((x1-x0)/2)
            dy = abs((y1-y0)/2)
            dz = abs(zc-zr)
            # Plot 3D bounding box
            X0 = xc-dx 
            X1 = xc+dx 
            Y0 = yc-dy 
            Y1 = yc+dy
            Z0 = zc-dz
            Z1 = zc+dz
            ax.plot([X0, X1, X1, X0, X0], [Y0, Y0, Y1, Y1, Y0], [Z0, Z0, Z0, Z0, Z0], color='r')
            ax.plot([X0, X1, X1, X0, X0], [Y0, Y0, Y1, Y1, Y0], [Z1, Z1, Z1, Z1, Z1], color='r')
            ax.plot([X0, X0], [Y0, Y0], [Z0, Z1], color='r')
            ax.plot([X0, X0], [Y1, Y1], [Z0, Z1], color='r')
            ax.plot([X1, X1], [Y0, Y0], [Z0, Z1], color='r')
            ax.plot([X1, X1], [Y1, Y1], [Z0, Z1], color='r')

        # Save plot
        plt.savefig(f"outputs/boxes/{filename}")
        plt.close()