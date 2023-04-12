# Source: https://learnopencv.com/faster-r-cnn-object-detection-with-pytorch/

import torchvision
from PIL import Image
from torchvision import transforms as T
import cv2
import matplotlib.pyplot as plt
import os

# Function to get the bounding boxes and labels of detected objects
def get_prediction(img_path, threshold, model, categories):
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img])
    pred_class = [categories[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x>threshold]
    if len(pred_t) == 0:
        return [], []
    pred_t = pred_t[-1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return pred_boxes, pred_class

# Function to plot bounding boxes around cars in each test image
def object_detection_api(directory, model, categories, threshold=0.5, rect_th=2, text_size=1, text_th=1): 
    for filename in os.listdir(directory):
        # print(filename)
        img_path = directory + filename
        boxes, pred_cls = get_prediction(img_path, threshold, model, categories) 
        img = cv2.imread(img_path) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        for i in range(len(boxes)):
            if pred_cls[i] == 'car':
                cv2.rectangle(img, (int(boxes[i][0][0]), int(boxes[i][0][1])), (int(boxes[i][1][0]), int(boxes[i][1][1])), color=(0,255,0), thickness=rect_th) 
                # cv2.putText(img,pred_cls[i], (int(boxes[i][0][0]), int(boxes[i][0][1])), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0), thickness=text_th) 
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('outputs/objects/' + filename, img)

if __name__ == "__main__":
    # Load the model
    COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # Detect cars in test images
    directory = 'data/test/image_left/'
    object_detection_api(directory, model, COCO_INSTANCE_CATEGORY_NAMES)