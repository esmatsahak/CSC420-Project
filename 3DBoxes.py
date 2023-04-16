import torchvision
import cv2 as cv
import numpy as np
import json
import matplotlib.pyplot as plt
import os
from objectDetection import get_prediction
from depth import computeDisparity, computeDepth, getCameraParams
from plane import getRoadPixels, preprocessRoadPixels, getBestPlane, plot3DPoints, convert2DTo3D

# Function to construct rotation matrix to rotate box to be parallel to road plane
def getRotationMatrix(plane_normal):
    v1 = np.cross(plane_normal, [1, 0, 0])
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(plane_normal, v1)
    v2 /= np.linalg.norm(v2)
    R = np.vstack((v1, v2, plane_normal))
    return R

if __name__ == '__main__':
    # Load object detection model
    COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # Iterate over test images
    directory = "data/test/image_left/"
    for i, filename in enumerate(os.listdir(directory)):
        # Load image
        image_id = filename.split(".")[0]
        # print(image_id)
        img_path = f"data/test/image_left/{image_id}.jpg"
        left_image = cv.imread(img_path)
        right_image = cv.imread(f"data/test/image_right/{image_id}.jpg")

        # Get depth map of image
        f, T, px, py = getCameraParams(f"data/test/calib/{image_id}.txt")
        disparity = computeDisparity(left_image, right_image, 64, 9, 1)
        depth = computeDepth(disparity, T, f)

        # Get road plane
        with open("outputs/road_dict.json", "r") as file:
            road_dict = json.load(file)
        road_pixels = getRoadPixels(left_image, road_dict, image_id)
        road_points = preprocessRoadPixels(road_pixels, depth, f, px, py)  
        plane, _ = getBestPlane(road_points)  

        # Compute rotation matrix to rotate box to be parallel to road plane
        a, b, c = plane[0], plane[1], plane[2]
        n = np.array([a, b, -1])      
        R = getRotationMatrix(n)

        # Get bounding boxes of cars in image
        boxes, pred_cls = get_prediction(img_path, 0.5, model, COCO_INSTANCE_CATEGORY_NAMES)
        car_idx = np.where(np.array(pred_cls) == 'car')
        car_boxes = np.array(boxes)[car_idx]

        # Plot road points and road plane
        ax = plot3DPoints(road_points, plane)

        # Iterate over bounding boxes
        for box in car_boxes:
            # Get bottom-left and top-right corners of bounding box
            y0, y1 = int(box[0][0]), int(box[1][0])
            x0, x1 = int(box[0][1]), int(box[1][1])
            # Get 3D coordinates of center of bounding box
            xc, yc = (x0 + x1)//2, (y0 + y1)//2
            zc = depth[xc, yc]
            Xc, Yc, Zc = convert2DTo3D(xc, yc, zc, f, px, py)
            min_X, min_Y, min_Z = float('inf'), float('inf'), float('inf')
            max_X, max_Y, max_Z = float('-inf'), float('-inf'), float('-inf')
            # Iterate over pixels in bounding box
            for x in range(x0, x1):
                for y in range(y0, y1):
                    # Get depth of pixel
                    z = depth[x,y]
                    # Get 3D representation of pixel in world coordinates
                    X, Y, Z = convert2DTo3D(x, y, z, f, px, py)
                    # Update min and max values
                    if np.linalg.norm(np.array([X, Y, Z]) - np.array([Xc, Yc, Zc])) < 30:
                        min_X, min_Y, min_Z = min(min_X, X), min(min_Y, Y), min(min_Z, Z)
                        max_X, max_Y, max_Z = max(max_X, X), max(max_Y, Y), max(max_Z, Z)
            # Get coordinates of 3D bounding box
            p = np.array([
                [min_X, min_Y, min_Z], 
                [min_X, max_Y, min_Z], 
                [max_X, min_Y, min_Z], 
                [max_X, max_Y, min_Z], 
                [min_X, min_Y, max_Z], 
                [min_X, max_Y, max_Z], 
                [max_X, min_Y, max_Z], 
                [max_X, max_Y, max_Z]]
            )
            # Rotate 3D bounding box to align with road plane
            pc = p.mean(axis=0)
            p -= pc         # Translate 3D bounding box to origin
            p = p.dot(R)
            p += pc         # Translate 3D bounding box back to original position
            # Translate 3D bounding box to lie on top of road plane
            min_point = p[np.argmin(p[:, 2])]
            min_Z_plane = np.dot(plane, [min_point[0], min_point[1], 1])
            Z_diff = abs(min_point[2] - min_Z_plane)
            if min_Z_plane > min_point[2]:
                p[:, 2] += Z_diff
            else:
                p[:, 2] -= Z_diff
            # Plot 3D bounding box   
            ax.plot([p[0,0], p[2,0], p[3,0], p[1,0], p[0,0]], 
                    [p[0,1], p[2,1], p[3,1], p[1,1], p[0,1]], 
                    [p[0,2], p[2,2], p[3,2], p[1,2], p[0,2]], color='r')
            ax.plot([p[4,0], p[6,0], p[7,0], p[5,0], p[4,0]], 
                    [p[4,1], p[6,1], p[7,1], p[5,1], p[4,1]], 
                    [p[4,2], p[6,2], p[7,2], p[5,2], p[4,2]], color='r')
            ax.plot([p[0,0], p[4,0]], [p[0,1], p[4,1]], [p[0,2], p[4,2]], color='r')
            ax.plot([p[1,0], p[5,0]], [p[1,1], p[5,1]], [p[1,2], p[5,2]], color='r')
            ax.plot([p[2,0], p[6,0]], [p[2,1], p[6,1]], [p[2,2], p[6,2]], color='r')
            ax.plot([p[3,0], p[7,0]], [p[3,1], p[7,1]], [p[3,2], p[7,2]], color='r')

        # Save plot
        plt.savefig(f"outputs/boxes/{filename}")
        plt.close()  