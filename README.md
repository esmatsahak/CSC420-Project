# CSC420-Project
This repo contains the code for for CSC420 Winter 2023 Project 2. Description of files with their corresponding objectives is as follows: 

### Disparity and Depth Maps
* `depth.py` contains the necessary functions for computing the disparity and depth maps of an input image. 
* `show_depth_maps.ipynb` contains the code and visualization for the maps for a sample image. 

### Road Features and Classifier
* `roadFeatures.py` contains the code for computing the features used by the MLP classifier in  `roadClassifier.py` to detect road pixels
and visualize predictions. 
* `road_classifier_unet.ipynb` contains the code for training and evaluating an Attention U-Net-based deep learning model for road segmentation. 
The final trained model can be found under `outputs/best_metric_model_final.pth`. 

### Ground Plane Detection 
* `plane.py` approximates the ground (road) plane of an image given the predicted road pixels. 
* `3DPointCloud.py` visualizes the results of the ground plane approximation in a 3D point cloud. 

### Car Detection
* `objectDetection.py` uses a pre-trained model to detect cars in an image and visualizes results with 2D bounding boxes. 

### Viewpoint Classifier 
* `viewpointFeatures.py` and `viewpointClassifier.py` train a classifier to determine the car orientation (viewpoint). 
* `visualization.py` visualizes the resulting predictions with viewpoint vectors.

### 3D Bounding Box
* `3DBoxes.py` computes 3D bounding box coordinates of detected cars and visualizes results in 3D plot with estimated ground plane. 
